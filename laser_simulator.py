"""
laser_simulator.py
==================
Laser-piirturin / CNC-kaivertimen G-koodi-simulaattori.

Ominaisuudet:
  - G-koodin jäsennys ja animoitu toisto
  - Realistinen polttojälki: säteily, hehku, kerrostuvuus
  - Zoom, panorointi (drag), mittaruudukko
  - Säädettävä nopeus, polton voimakkuus, radan näyttö
  - Askeleen eteen/taakse, suorahyppy riville
  - Tilaviiva: koordinaatit, teho, edistyminen
  - Vientimahdollisuus: tuloskuva PNG:ksi

Käynnistys:
  python laser_simulator.py
  python laser_simulator.py tiedosto.gcode

Riippuvuudet:
  PyQt5, numpy  (opencv ei tarvita simulaattorissa)
"""

import sys
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QStatusBar,
    QToolBar, QAction, QSizePolicy, QProgressBar, QFrame,
    QGridLayout, QMessageBox,
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QPointF, QRectF,
    QElapsedTimer,
)
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QFontMetrics,
    QImage, QWheelEvent, QMouseEvent, QKeyEvent,
)


# ===========================================================================
# G-KOODIN JÄSENNYS
# ===========================================================================

@dataclass
class GCmd:
    """Yksi jäsennetty G-koodi-käsky."""
    line_num: int        # Rivinumero tiedostossa (1-pohjainen)
    raw: str             # Alkuperäinen tekstirivi
    cmd: str             # Pääkäsky isolla, esim. "G1"
    params: dict         # Parametrit, esim. {"X": 10.5, "S": 1000}


def parse_gcode(text: str) -> list[GCmd]:
    """
    Jäsentää G-kooditekstin GCmd-listaksi.

    - Kommentit (; jälkeen) poistetaan.
    - Tyhjät rivit ohitetaan.
    - Parametrit (kirjain + numero) poimitaan sanakirjaan.
    """
    cmds = []
    for li, raw in enumerate(text.splitlines(), start=1):
        clean = raw.split(";")[0].strip().upper()
        if not clean:
            continue
        tokens = clean.split()
        if not tokens:
            continue
        main = tokens[0]
        params = {}
        for tok in tokens[1:]:
            m = re.match(r"^([A-Z])([-+]?\d*\.?\d+)$", tok)
            if m:
                params[m.group(1)] = float(m.group(2))
        cmds.append(GCmd(line_num=li, raw=raw.rstrip(), cmd=main, params=params))
    return cmds


# ===========================================================================
# SIMULAATTORIN TILA
# ===========================================================================

@dataclass
class SimState:
    """Simulaattorin fyysinen tila yhden käskyn jälkeen."""
    x: float = 0.0          # Nykyinen X mm
    y: float = 0.0          # Nykyinen Y mm
    power: float = 0.0      # Laserin teho 0–max_power
    max_power: float = 1000.0
    abs_mode: bool = True    # G90=abs, G91=rel
    feed: float = 1000.0     # Syöttönopeus mm/min
    laser_on: bool = False   # M3/M5 tila


@dataclass
class Segment:
    """
    Yksi piirto- tai siirtosegmentti.
    Tallennetaan etukäteen koko radan esikäsittelyä varten.
    """
    x0: float
    y0: float
    x1: float
    y1: float
    power: float   # 0 = siirto, >0 = poltto
    feed: float
    cmd_idx: int   # Indeksi commands-listaan


def preprocess(cmds: list[GCmd]) -> tuple[list[Segment], float]:
    """
    Muuntaa GCmd-listan Segment-listaksi.

    Palauttaa (segmentit, max_power).
    Segmentit ovat järjestyksessä; siirrot (G0) tallennetaan power=0:lla.
    """
    segments: list[Segment] = []
    s = SimState()

    # Etsi ensin max_power kaikista M3-käskyistä
    max_p = 1.0
    for c in cmds:
        if c.cmd == "M3" and "S" in c.params:
            max_p = max(max_p, c.params["S"])
    s.max_power = max_p

    for i, c in enumerate(cmds):
        p = c.params

        if c.cmd == "G20":
            # Tuumamoodi – ei tueta, jatketaan millimetreissä
            print(f"VAROITUS rivi {c.line_num}: G20 (tuumamoodi) ei ole tuettu, "
                  f"käytetään millimetrejä.")
        elif c.cmd == "G21":
            pass  # Millimetrimoodi – oletus, ei tehdä mitään
        elif c.cmd == "G90":
            s.abs_mode = True
        elif c.cmd == "G91":
            s.abs_mode = False
        elif c.cmd == "G28":
            # Kotiasema: siirto origoon, laser pois päältä
            segments.append(Segment(s.x, s.y, 0.0, 0.0, 0.0, s.feed, i))
            s.x, s.y = 0.0, 0.0
            s.laser_on = False
            s.power = 0
        elif c.cmd == "M3":
            if "S" in p:
                s.power = p["S"]
            s.laser_on = True
        elif c.cmd in ("M5", "M2", "M30"):
            s.power = 0; s.laser_on = False
        elif c.cmd in ("G0", "G1"):
            if "F" in p:
                s.feed = p["F"]
            nx = (p["X"] if s.abs_mode else s.x + p["X"]) if "X" in p else s.x
            ny = (p["Y"] if s.abs_mode else s.y + p["Y"]) if "Y" in p else s.y
            burn = (c.cmd == "G1") and s.laser_on and s.power > 0
            pw = s.power if burn else 0.0
            segments.append(Segment(s.x, s.y, nx, ny, pw, s.feed, i))
            s.x, s.y = nx, ny
        elif c.cmd == "G4":
            # Dwell: nollapituinen segmentti paikan merkiksi
            segments.append(Segment(s.x, s.y, s.x, s.y, s.power if s.laser_on else 0, s.feed, i))

    return segments, max_p


# ===========================================================================
# PIIRTOKANVAS – QWidget jossa QPainter-renderöinti
# ===========================================================================

class SimCanvas(QWidget):
    """
    Simulaattorin pääpiirtoalue.

    Renderöintistrategia:
      1. burn_image (QImage, ARGB32) – kumulatiivinen polttojälki,
         piirretään inkrementaalisesti sitä mukaa kuin segmentit valmistuvat.
      2. Jokainen frame:
         a) täytetään tausta (tumma)
         b) piirretään burn_image
         c) piirretään mittaruudukko (semi-transparent)
         d) piirretään lasersäteen "hehku" nykyisestä pisteestä
         e) piirretään laserpään ristikko
         f) piirretään tilainfo

    Zoom ja panorointi toteutetaan QTransform-muunnoksella.
    """

    # Signaali: koordinaatit mousemovella (mm)
    mouse_mm = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # --- Simulaattorin data ---
        self.segments: list[Segment] = []
        self.current_seg_idx: int = 0   # Kuinka monta segmenttiä on piirretty
        self.max_power: float = 1000.0
        self.area_w_mm: float = 200.0   # Piirtoalueen leveys mm
        self.area_h_mm: float = 150.0   # Piirtoalueen korkeus mm

        # Nykyinen laserpään sijainti ja tila
        self.head_x: float = 0.0
        self.head_y: float = 0.0
        self.head_power: float = 0.0

        # --- Grafiikka-asetukset ---
        self.show_travel = True         # Näytä siirtoreitit
        self.show_grid = True           # Näytä ruudukko
        self.show_head = True           # Näytä laserpää
        self.burn_intensity = 1.0       # Polttovoimakkuuskerroin (0.1–2.0)
        self.glow_radius = 18           # Hehkusäde pikseleinä

        # --- Zoom ja panorointi ---
        self._zoom = 1.0
        self._pan_x = 0.0   # Panorointisiirto pikseleinä
        self._pan_y = 0.0
        self._drag_start = None
        self._pan_start = None

        # --- Polttokuva ---
        # _burn_img luodaan _init_burn_image():ssa ja piirretään resizeEvent:ssä
        self._burn_img: Optional[QImage] = None
        self._img_w = 0
        self._img_h = 0

        # Edellinen piirretty segmentti-indeksi (inkrementaalista piirtoa varten)
        self._last_drawn_seg = 0

        self._init_burn_image()

    # ------------------------------------------------------------------
    # Julkinen API
    # ------------------------------------------------------------------

    def redraw_all(self):
        """
        Piirtää kaikki tähänastiset segmentit uudelleen tyhjältä pohjalta.
        Käytetään kun visuaalinen asetus muuttuu (voimakkuus, siirtoreittien näkyvyys).
        Julkinen metodi – kutsujien ei tarvitse koskea _last_drawn_seg:iin suoraan.
        """
        saved_idx = self._last_drawn_seg
        self._init_burn_image()
        self._last_drawn_seg = 0
        self.advance_to(saved_idx)

    def reset(self):
        """Nollaa polttokuvan ja simulaattorin tilan."""
        self.current_seg_idx = 0
        self._last_drawn_seg = 0
        self.head_x = self.head_y = self.head_power = 0.0
        self._init_burn_image()
        self.update()

    def set_segments(self, segs: list[Segment], max_power: float,
                     area_w: float, area_h: float):
        """Asettaa uuden segmenttilistan ja nollaa tilan."""
        self.segments = segs
        self.max_power = max(1.0, max_power)
        self.area_w_mm = area_w
        self.area_h_mm = area_h
        self.reset()
        self._fit_view()

    def advance_to(self, seg_idx: int):
        """
        Piirtää polttokuvaan kaikki segmentit indeksiin seg_idx asti.
        Inkrementaalinen: jos seg_idx > _last_drawn_seg, piirretään vain uudet.
        Jos seg_idx < _last_drawn_seg (taaksepäin), piirretään kaikki uudelleen.
        """
        if seg_idx < self._last_drawn_seg:
            # Taaksepäin → piirrä kaikki uudelleen
            self._init_burn_image()
            self._last_drawn_seg = 0

        if seg_idx <= self._last_drawn_seg:
            self.current_seg_idx = seg_idx
            if self.segments and seg_idx > 0:
                s = self.segments[seg_idx - 1]
                self.head_x, self.head_y = s.x1, s.y1
                self.head_power = s.power
            self.update()
            return

        # Piirrä uudet segmentit polttokuvaan
        painter = QPainter(self._burn_img)
        painter.setRenderHint(QPainter.Antialiasing, True)

        for i in range(self._last_drawn_seg, seg_idx):
            if i >= len(self.segments):
                break
            seg = self.segments[i]
            self._draw_segment_to_burn(painter, seg)

        painter.end()

        self._last_drawn_seg = seg_idx
        self.current_seg_idx = seg_idx

        if self.segments and seg_idx > 0:
            s = self.segments[seg_idx - 1]
            self.head_x, self.head_y = s.x1, s.y1
            self.head_power = s.power

        self.update()

    def export_image(self, path: str):
        """Vie polttokuvan PNG-tiedostoon."""
        if self._burn_img:
            self._burn_img.save(path)

    def fit_view(self):
        """Sovittaa näkymän piirtoalueelle."""
        self._fit_view()
        self.update()

    # ------------------------------------------------------------------
    # Sisäiset apumetodit
    # ------------------------------------------------------------------

    def _init_burn_image(self):
        """Luo tai tyhjentää polttokuvan (musta tausta)."""
        w = max(1, self.width())
        h = max(1, self.height())
        self._img_w, self._img_h = w, h
        self._burn_img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        self._burn_img.fill(QColor(10, 8, 6))  # Lähes musta puupohja

    def _fit_view(self):
        """Laskee zoom ja pan niin että piirtoalue näkyy kokonaan."""
        if self.width() <= 0 or self.height() <= 0:
            return
        margin = 40  # pikseleinä
        aw, ah = self.width() - 2 * margin, self.height() - 2 * margin
        sx = aw / max(0.01, self.area_w_mm)
        sy = ah / max(0.01, self.area_h_mm)
        self._zoom = min(sx, sy)
        # Centroi piirtoalue
        drawn_w = self.area_w_mm * self._zoom
        drawn_h = self.area_h_mm * self._zoom
        self._pan_x = (self.width() - drawn_w) / 2
        self._pan_y = (self.height() - drawn_h) / 2

    def _mm_to_screen(self, xmm: float, ymm: float) -> QPointF:
        """Muuntaa mm-koordinaatit ruutukoordinaateiksi."""
        sx = xmm * self._zoom + self._pan_x
        # Y käännetään: mm:ssä Y kasvaa ylöspäin, ruudulla alas
        sy = (self.area_h_mm - ymm) * self._zoom + self._pan_y
        return QPointF(sx, sy)

    def _screen_to_mm(self, sx: float, sy: float) -> tuple:
        """Muuntaa ruutukoordinaatit mm-koordinaateiksi."""
        xmm = (sx - self._pan_x) / max(0.001, self._zoom)
        ymm = self.area_h_mm - (sy - self._pan_y) / max(0.001, self._zoom)
        return xmm, ymm

    def _draw_segment_to_burn(self, painter: QPainter, seg: Segment):
        """
        Piirtää yhden segmentin polttokuvaan.

        Polttosegmentti: useita kerroksia hehkua + ydinviiva.
        Siirtosegmentti: ohut katkoviiva (jos show_travel on päällä).
        """
        p0 = self._mm_to_screen(seg.x0, seg.y0)
        p1 = self._mm_to_screen(seg.x1, seg.y1)

        if seg.power > 0:
            # Normalisoitu teho 0–1
            rel = min(1.0, seg.power / self.max_power) * self.burn_intensity

            # Kerros 1: leveä oranssi hehku (ulompi, läpikuultava)
            glow_w = max(1.5, self.glow_radius * rel * 0.6)
            pen = QPen(QColor(255, 80, 0, int(30 * rel)))
            pen.setWidthF(glow_w * 2.5)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(p0, p1)

            # Kerros 2: keltainen hehku (sisempi)
            pen2 = QPen(QColor(255, 180, 20, int(60 * rel)))
            pen2.setWidthF(glow_w * 1.2)
            pen2.setCapStyle(Qt.RoundCap)
            painter.setPen(pen2)
            painter.drawLine(p0, p1)

            # Kerros 3: tumma "palanut" ura – ENNEN ydinviivaa jotta ydin
            # näkyy päälle eikä peity tumman alle
            core_w = max(0.4, 1.2 * rel)
            pen4 = QPen(QColor(20, 12, 4, int(180 * rel)))
            pen4.setWidthF(max(0.5, core_w * 0.7))
            pen4.setCapStyle(Qt.RoundCap)
            painter.setPen(pen4)
            painter.drawLine(p0, p1)

            # Kerros 4: valkoinen ydinviiva päällimmäisenä (kirkkain)
            pen3 = QPen(QColor(255, 240, 200, int(200 * rel)))
            pen3.setWidthF(core_w)
            pen3.setCapStyle(Qt.RoundCap)
            painter.setPen(pen3)
            painter.drawLine(p0, p1)

        elif self.show_travel:
            # Siirtoreitti: ohut sininen katkoviiva
            pen = QPen(QColor(60, 120, 255, 55))
            pen.setWidthF(0.7)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(p0, p1)

    # ------------------------------------------------------------------
    # Qt-tapahtumat
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Luodaan uusi polttokuva uudessa koossa ja piirretään kaikki uudelleen
        self._init_burn_image()
        old_last = self._last_drawn_seg
        self._last_drawn_seg = 0
        if old_last > 0:
            # Piirrä kaikki segmentit uudelleen
            p = QPainter(self._burn_img)
            p.setRenderHint(QPainter.Antialiasing, True)
            for i in range(min(old_last, len(self.segments))):
                self._draw_segment_to_burn(p, self.segments[i])
            p.end()
            self._last_drawn_seg = old_last
        self._fit_view()

    def paintEvent(self, event):
        """
        Päärenderöinti joka framella:
          1. Tausta
          2. Polttokuva
          3. Ruudukko
          4. Piirtoalueen reunat
          5. Laserpää + hehku
          6. Koordinaattiakselit
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1. Tausta
        painter.fillRect(self.rect(), QColor(18, 14, 10))

        # 2. Polttokuva
        if self._burn_img:
            painter.drawImage(0, 0, self._burn_img)

        # 3. Ruudukko
        if self.show_grid:
            self._paint_grid(painter)

        # 4. Piirtoalueen reunakehys
        self._paint_border(painter)

        # 5. Alkupiste (origo)
        self._paint_origin(painter)

        # 6. Laserpää
        if self.show_head:
            self._paint_laser_head(painter)

        # 7. Asteikot reunoilla
        self._paint_rulers(painter)

        painter.end()

    def _paint_grid(self, painter: QPainter):
        """Piirtää millimetriruudukon."""
        # Laske mikä mm-väli on järkevä zoomtasolla
        # Tavoite: noin 40–80px välein
        target_px = 50
        mm_step_candidates = [1, 2, 5, 10, 20, 50, 100]
        mm_step = 10
        for s in mm_step_candidates:
            if s * self._zoom >= target_px:
                mm_step = s
                break

        pen_minor = QPen(QColor(255, 255, 255, 18))
        pen_minor.setWidthF(0.5)
        pen_major = QPen(QColor(255, 255, 255, 35))
        pen_major.setWidthF(0.5)

        x0mm = max(0, int(-self._pan_x / self._zoom / mm_step) * mm_step - mm_step)
        y0mm = max(0, int(-self._pan_y / self._zoom / mm_step) * mm_step - mm_step)
        x1mm = x0mm + self.width() / self._zoom + mm_step * 2
        y1mm = y0mm + self.height() / self._zoom + mm_step * 2

        xm = x0mm
        while xm <= x1mm:
            p = self._mm_to_screen(xm, 0)
            is_major = (xm % (mm_step * 5) == 0)
            painter.setPen(pen_major if is_major else pen_minor)
            painter.drawLine(QPointF(p.x(), 0), QPointF(p.x(), self.height()))
            xm += mm_step

        ym = y0mm
        while ym <= y1mm:
            p = self._mm_to_screen(0, ym)
            is_major = (ym % (mm_step * 5) == 0)
            painter.setPen(pen_major if is_major else pen_minor)
            painter.drawLine(QPointF(0, p.y()), QPointF(self.width(), p.y()))
            ym += mm_step

    def _paint_border(self, painter: QPainter):
        """Piirtää piirtoalueen reunat."""
        tl = self._mm_to_screen(0, self.area_h_mm)
        br = self._mm_to_screen(self.area_w_mm, 0)
        rect = QRectF(tl, br)
        pen = QPen(QColor(80, 160, 255, 130))
        pen.setWidthF(1.5)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

    def _paint_origin(self, painter: QPainter):
        """Piirtää origon koordinaattiristikon."""
        o = self._mm_to_screen(0, 0)
        size = 12
        pen = QPen(QColor(100, 220, 100, 180))
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.drawLine(QPointF(o.x() - size, o.y()), QPointF(o.x() + size, o.y()))
        painter.drawLine(QPointF(o.x(), o.y() - size), QPointF(o.x(), o.y() + size))
        # Pieni ympyrä origossa
        painter.setBrush(QBrush(QColor(100, 220, 100, 120)))
        painter.drawEllipse(o, 3.5, 3.5)

    def _paint_laser_head(self, painter: QPainter):
        """Piirtää laserpään: ristikko + hehkuefekti jos laser päällä."""
        pos = self._mm_to_screen(self.head_x, self.head_y)
        rel_power = min(1.0, self.head_power / max(1, self.max_power))
        laser_on = self.head_power > 0

        if laser_on:
            # Ulompi hehkurengas (radiaaligradientti)
            grad_r = self.glow_radius * (0.5 + rel_power * 0.5)
            for r, alpha, color in [
                (grad_r * 2.0, int(15 * rel_power), QColor(255, 60, 0)),
                (grad_r * 1.2, int(40 * rel_power), QColor(255, 140, 0)),
                (grad_r * 0.6, int(90 * rel_power), QColor(255, 220, 100)),
            ]:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(color.red(), color.green(),
                                              color.blue(), alpha)))
                painter.drawEllipse(pos, r, r)

            # Ytimen piste
            core_r = max(2.5, 5 * rel_power)
            painter.setBrush(QBrush(QColor(255, 255, 240, 230)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(pos, core_r, core_r)

        # Ristikko (näkyy aina)
        cross = 14
        pen_cross = QPen(QColor(0, 200, 255, 200) if not laser_on
                         else QColor(255, 220, 0, 220))
        pen_cross.setWidthF(1.2)
        painter.setPen(pen_cross)
        painter.drawLine(QPointF(pos.x() - cross, pos.y()),
                         QPointF(pos.x() + cross, pos.y()))
        painter.drawLine(QPointF(pos.x(), pos.y() - cross),
                         QPointF(pos.x(), pos.y() + cross))

        # Ympyrä ristikon ympärillä
        circle_r = 7.0
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(pos, circle_r, circle_r)

    def _paint_rulers(self, painter: QPainter):
        """Piirtää millimetriasteikot vasempaan ja alareunaan."""
        target_px = 50
        mm_step_candidates = [1, 2, 5, 10, 20, 50, 100]
        mm_step = 10
        for s in mm_step_candidates:
            if s * self._zoom >= target_px:
                mm_step = s
                break

        font = QFont("monospace", 9)
        painter.setFont(font)
        fm = QFontMetrics(font)

        pen = QPen(QColor(180, 180, 180, 120))
        pen.setWidthF(0.7)
        painter.setPen(pen)

        # X-akseli (alareuna)
        y_ruler = self.height() - 18
        xm = 0
        while xm <= self.area_w_mm + mm_step:
            p = self._mm_to_screen(xm, 0)
            if 0 < p.x() < self.width():
                tick_h = 8 if xm % (mm_step * 5) == 0 else 4
                painter.drawLine(QPointF(p.x(), y_ruler),
                                 QPointF(p.x(), y_ruler + tick_h))
                if xm % (mm_step * 5) == 0:
                    lbl = str(int(xm))
                    tw = fm.horizontalAdvance(lbl)
                    painter.drawText(QPointF(p.x() - tw / 2, y_ruler + tick_h + 11),
                                     lbl)
            xm += mm_step

        # Y-akseli (vasen reuna)
        x_ruler = 16
        ym = 0
        while ym <= self.area_h_mm + mm_step:
            p = self._mm_to_screen(0, ym)
            if 0 < p.y() < self.height():
                tick_w = 8 if ym % (mm_step * 5) == 0 else 4
                painter.drawLine(QPointF(x_ruler, p.y()),
                                 QPointF(x_ruler + tick_w, p.y()))
                if ym % (mm_step * 5) == 0:
                    lbl = str(int(ym))
                    th = fm.height()
                    painter.drawText(QPointF(2, p.y() + th / 3), lbl)
            ym += mm_step

    # ------------------------------------------------------------------
    # Zoom ja panorointi
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent):
        """Zoomataan hiiren rullan suuntaan kursorin kohtaan."""
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 1 / 1.12
        mx, my = event.pos().x(), event.pos().y()

        # Pidä kursorin alla oleva mm-koordinaatti paikallaan
        xmm, ymm = self._screen_to_mm(mx, my)
        self._zoom *= factor
        self._zoom = max(0.1, min(self._zoom, 200.0))

        # Laske uusi pan niin että sama mm-kohta pysyy kursorin alla
        p = self._mm_to_screen(xmm, ymm)
        self._pan_x += mx - p.x()
        self._pan_y += my - p.y()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton or \
           (event.button() == Qt.LeftButton and
            event.modifiers() & Qt.AltModifier):
            self._drag_start = event.pos()
            self._pan_start = (self._pan_x, self._pan_y)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        xmm, ymm = self._screen_to_mm(event.pos().x(), event.pos().y())
        self.mouse_mm.emit(xmm, ymm)

        if self._drag_start is not None:
            dx = event.pos().x() - self._drag_start.x()
            dy = event.pos().y() - self._drag_start.y()
            self._pan_x = self._pan_start[0] + dx
            self._pan_y = self._pan_start[1] + dy
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Lopettaa panoroinnin vain kun oikea painike vapautetaan."""
        if self._drag_start is not None and (
            event.button() == Qt.MiddleButton or
            event.button() == Qt.LeftButton
        ):
            self._drag_start = None
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event: QKeyEvent):
        """Näppäinoikopolut: F = sovita, + / - = zoom."""
        if event.key() == Qt.Key_F:
            self._fit_view()
            self.update()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            cx, cy = self.width() / 2, self.height() / 2
            xmm, ymm = self._screen_to_mm(cx, cy)
            self._zoom *= 1.2
            p = self._mm_to_screen(xmm, ymm)
            self._pan_x += cx - p.x(); self._pan_y += cy - p.y()
            self.update()
        elif event.key() == Qt.Key_Minus:
            cx, cy = self.width() / 2, self.height() / 2
            xmm, ymm = self._screen_to_mm(cx, cy)
            self._zoom /= 1.2
            p = self._mm_to_screen(xmm, ymm)
            self._pan_x += cx - p.x(); self._pan_y += cy - p.y()
            self.update()


# ===========================================================================
# PÄÄIKKUNA
# ===========================================================================

class SimulatorWindow(QMainWindow):
    """
    Simulaattorin pääikkuna.

    Vasemmalla: ohjauspaneeli (tiedosto, toisto, asetukset).
    Oikealla:   SimCanvas-piirtoalue.
    Alhaalla:   tilaviiva ja edistymispalkki.
    """

    def __init__(self, gcode_path: str = None):
        super().__init__()
        self.setWindowTitle("Laser-simulaattori")
        self.resize(1280, 780)

        self.commands: list[GCmd] = []
        self.segments: list[Segment] = []
        self.max_power = 1000.0

        self._playing = False
        self._current_seg = 0   # Kuinka monta segmenttiä on näytetty

        # Timer animaatioon
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._tick)
        self._steps_per_tick = 5    # Segmenttiä per frame (= nopeus)

        self._elapsed = QElapsedTimer()
        self._elapsed_ms_accumulated: int = 0  # Kumulatiivinen aika ms pausen yli

        # Tallennushakemisto "Output" ohjelman omassa kansiossa
        self.output_dir = Path(__file__).resolve().parent / "Output"
        self.output_dir.mkdir(exist_ok=True)

        self._build_ui()
        self._apply_style()

        # Lataa tiedosto käynnistysargumentista
        if gcode_path and Path(gcode_path).exists():
            self._load_file(gcode_path)

    # ------------------------------------------------------------------
    # UI-RAKENNE
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Rakentaa ikkunan layout-rakenteen."""
        # Toolbar
        tb = QToolBar("Toiminnot", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("📂  Avaa G-koodi", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_file)
        tb.addAction(act_open)

        tb.addSeparator()

        act_play = QAction("▶  Käynnistä", self)
        act_play.setShortcut("Space")
        act_play.triggered.connect(self._play)
        tb.addAction(act_play)

        act_pause = QAction("⏸  Tauko", self)
        # Ei omaa pikanäppäintä: Space hoituu keyPressEvent:ssä toggle-logiikalla
        act_pause.triggered.connect(self._pause)
        tb.addAction(act_pause)

        act_reset = QAction("⏮  Nollaa", self)
        act_reset.setShortcut("R")
        act_reset.triggered.connect(self._reset_sim)
        tb.addAction(act_reset)

        tb.addSeparator()

        act_step_fwd = QAction("⏭  +1 askel", self)
        act_step_fwd.setShortcut("Right")
        act_step_fwd.triggered.connect(self._step_forward)
        tb.addAction(act_step_fwd)

        act_step_bwd = QAction("⏮  −1 askel", self)
        act_step_bwd.setShortcut("Left")
        act_step_bwd.triggered.connect(self._step_backward)
        tb.addAction(act_step_bwd)

        tb.addSeparator()

        act_fit = QAction("⊡  Sovita", self)
        act_fit.setShortcut("F")
        act_fit.triggered.connect(self._fit_view)
        tb.addAction(act_fit)

        act_export = QAction("💾  Vie PNG", self)
        act_export.setShortcut("Ctrl+S")
        act_export.triggered.connect(self._export_png)
        tb.addAction(act_export)

        # --- Keskeiset widgetit ---
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Vasen paneeli
        left = self._build_left_panel()
        left.setMaximumWidth(260)
        root.addWidget(left)

        # Erotinviiva
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        # Canvas
        self.canvas = SimCanvas()
        self.canvas.mouse_mm.connect(self._on_mouse_mm)
        root.addWidget(self.canvas, stretch=1)

        # Statusbar
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self._lbl_pos = QLabel("X: 0.000  Y: 0.000 mm")
        self._lbl_power = QLabel("Teho: 0")
        self._lbl_seg = QLabel("0 / 0 segmenttiä")
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setMaximumHeight(14)
        self._progress.setTextVisible(False)
        sb.addWidget(self._lbl_pos)
        sb.addPermanentWidget(self._lbl_power)
        sb.addPermanentWidget(self._lbl_seg)
        sb.addPermanentWidget(self._progress)

    def _build_left_panel(self) -> QWidget:
        """Vasen ohjauspaneeli: tiedosto, toisto, asetukset."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Tiedosto ---
        grp_file = QGroupBox("Tiedosto")
        fl = QVBoxLayout(grp_file)
        btn_open = QPushButton("📂  Avaa .gcode")
        btn_open.clicked.connect(self._open_file)
        fl.addWidget(btn_open)
        self._lbl_filename = QLabel("Ei tiedostoa")
        self._lbl_filename.setWordWrap(True)
        self._lbl_filename.setStyleSheet("color: #aaa; font-size: 11px;")
        fl.addWidget(self._lbl_filename)
        layout.addWidget(grp_file)

        # --- Toisto ---
        grp_play = QGroupBox("Toisto")
        pl = QGridLayout(grp_play)

        btn_play = QPushButton("▶  Käynnistä")
        btn_play.setObjectName("btn_play")
        btn_play.clicked.connect(self._play)
        pl.addWidget(btn_play, 0, 0)

        btn_pause = QPushButton("⏸  Tauko")
        btn_pause.clicked.connect(self._pause)
        pl.addWidget(btn_pause, 0, 1)

        btn_reset = QPushButton("⏮  Nollaa")
        btn_reset.clicked.connect(self._reset_sim)
        pl.addWidget(btn_reset, 1, 0)

        btn_end = QPushButton("⏭  Loppu")
        btn_end.clicked.connect(self._jump_to_end)
        pl.addWidget(btn_end, 1, 1)

        pl.addWidget(QLabel("Nopeus (segm./frame):"), 2, 0, 1, 2)
        self._sld_speed = QSlider(Qt.Horizontal)
        self._sld_speed.setRange(1, 500)
        self._sld_speed.setValue(10)
        self._sld_speed.setTickInterval(50)
        self._sld_speed.setTickPosition(QSlider.TicksBelow)
        self._sld_speed.valueChanged.connect(self._on_speed_changed)
        pl.addWidget(self._sld_speed, 3, 0, 1, 2)
        self._lbl_speed = QLabel("10 segm./frame")
        self._lbl_speed.setAlignment(Qt.AlignCenter)
        pl.addWidget(self._lbl_speed, 4, 0, 1, 2)

        pl.addWidget(QLabel("Hyppää segmenttiin:"), 5, 0, 1, 2)
        jump_row = QHBoxLayout()
        self._spin_jump = QSpinBox()
        self._spin_jump.setRange(0, 0)
        jump_row.addWidget(self._spin_jump)
        btn_jump = QPushButton("Mene")
        btn_jump.clicked.connect(self._jump_to_segment)
        jump_row.addWidget(btn_jump)
        pl.addLayout(jump_row, 6, 0, 1, 2)

        layout.addWidget(grp_play)

        # --- Näyttöasetukset ---
        grp_vis = QGroupBox("Näyttö")
        vl = QVBoxLayout(grp_vis)

        self._chk_travel = QCheckBox("Näytä siirtoreitit")
        self._chk_travel.setChecked(True)
        self._chk_travel.toggled.connect(self._on_vis_changed)
        vl.addWidget(self._chk_travel)

        self._chk_grid = QCheckBox("Näytä ruudukko")
        self._chk_grid.setChecked(True)
        self._chk_grid.toggled.connect(lambda v: setattr(self.canvas, 'show_grid', v) or self.canvas.update())
        vl.addWidget(self._chk_grid)

        self._chk_head = QCheckBox("Näytä laserpää")
        self._chk_head.setChecked(True)
        self._chk_head.toggled.connect(lambda v: setattr(self.canvas, 'show_head', v) or self.canvas.update())
        vl.addWidget(self._chk_head)

        vl.addWidget(QLabel("Polton voimakkuus:"))
        self._sld_burn = QSlider(Qt.Horizontal)
        self._sld_burn.setRange(10, 250)
        self._sld_burn.setValue(100)
        self._sld_burn.valueChanged.connect(self._on_burn_changed)
        vl.addWidget(self._sld_burn)
        self._lbl_burn = QLabel("1.00×")
        self._lbl_burn.setAlignment(Qt.AlignCenter)
        vl.addWidget(self._lbl_burn)

        vl.addWidget(QLabel("Hehkun koko (px):"))
        self._sld_glow = QSlider(Qt.Horizontal)
        self._sld_glow.setRange(4, 60)
        self._sld_glow.setValue(18)
        self._sld_glow.valueChanged.connect(
            lambda v: setattr(self.canvas, 'glow_radius', v))
        vl.addWidget(self._sld_glow)

        layout.addWidget(grp_vis)

        # --- Piirtoalueen mitat ---
        grp_area = QGroupBox("Piirtoalue (mm)")
        al = QGridLayout(grp_area)
        al.addWidget(QLabel("Leveys:"), 0, 0)
        self._spin_aw = QDoubleSpinBox()
        self._spin_aw.setRange(10, 2000); self._spin_aw.setValue(200)
        self._spin_aw.valueChanged.connect(self._on_area_changed)
        al.addWidget(self._spin_aw, 0, 1)
        al.addWidget(QLabel("Korkeus:"), 1, 0)
        self._spin_ah = QDoubleSpinBox()
        self._spin_ah.setRange(10, 2000); self._spin_ah.setValue(150)
        self._spin_ah.valueChanged.connect(self._on_area_changed)
        al.addWidget(self._spin_ah, 1, 1)
        layout.addWidget(grp_area)

        # --- Vie PNG ---
        btn_export = QPushButton("💾  Vie kuva PNG:ksi")
        btn_export.clicked.connect(self._export_png)
        layout.addWidget(btn_export)

        layout.addStretch()

        # --- G-koodin tekstinäyttö ---
        grp_gc = QGroupBox("G-koodi (nykyinen rivi)")
        gcl = QVBoxLayout(grp_gc)
        self._lbl_gcline = QLabel("")
        self._lbl_gcline.setWordWrap(True)
        self._lbl_gcline.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #8fc; "
            "background: #111; padding: 4px; border-radius: 3px;"
        )
        gcl.addWidget(self._lbl_gcline)
        layout.addWidget(grp_gc)

        return panel

    # ------------------------------------------------------------------
    # TYYLI
    # ------------------------------------------------------------------

    def _apply_style(self):
        """Tumma teema sovellukselle."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1614;
                color: #d8d0c8;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #3a3028;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                color: #e8a060;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #e8a060;
            }
            QPushButton {
                background-color: #2a2018;
                border: 1px solid #4a3828;
                border-radius: 5px;
                padding: 5px 12px;
                color: #d8d0c8;
                min-height: 26px;
            }
            QPushButton:hover  { background-color: #3a3028; border-color: #e8a060; }
            QPushButton:pressed { background-color: #1a1208; }
            QPushButton#btn_play {
                background-color: #1a3a18;
                border-color: #4a8040;
                color: #a0e890;
            }
            QPushButton#btn_play:hover { background-color: #2a4a28; }
            QSlider::groove:horizontal {
                background: #2a2018; height: 5px; border-radius: 2px;
                border: 0.5px solid #3a3028;
            }
            QSlider::handle:horizontal {
                background: #e8a060; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #c06020; border-radius: 2px; }
            QSpinBox, QDoubleSpinBox {
                background: #2a2018; border: 1px solid #4a3828;
                border-radius: 4px; padding: 3px 6px; color: #d8d0c8;
            }
            QCheckBox { spacing: 6px; color: #c8c0b0; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #5a4838; border-radius: 3px;
                background: #2a2018;
            }
            QCheckBox::indicator:checked { background: #e8a060; border-color: #e8a060; }
            QLabel { color: #c8c0b0; }
            QToolBar {
                background: #141210; border-bottom: 1px solid #3a3028;
                spacing: 4px; padding: 3px;
            }
            QToolBar QToolButton {
                background: transparent; border: none;
                border-radius: 4px; padding: 4px 8px; color: #c8c0b0;
                font-size: 12px;
            }
            QToolBar QToolButton:hover { background: #3a3028; }
            QStatusBar { background: #141210; border-top: 1px solid #3a3028; }
            QStatusBar QLabel { color: #a09888; font-size: 11px; }
            QProgressBar {
                background: #2a2018; border: none; border-radius: 2px;
            }
            QProgressBar::chunk { background: #e8a060; }
            QFrame[frameShape="5"] { color: #3a3028; }
        """)

    # ------------------------------------------------------------------
    # TIEDOSTON LATAUS
    # ------------------------------------------------------------------

    def _open_file(self):
        """Avaa tiedostonvalintaikkuna ja lataa G-koodin."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Avaa G-koodi-tiedosto",
            str(Path(__file__).resolve().parent),
            "G-koodi (*.gcode *.nc *.txt *.gc);;Kaikki (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        """Lataa ja jäsentää G-kooditiedoston, nollaa simulaattorin."""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            QMessageBox.critical(self, "Latausvirhe", str(exc))
            return

        self.commands = parse_gcode(text)
        if not self.commands:
            QMessageBox.warning(self, "Tyhjä tiedosto",
                                "Tiedostossa ei ole G-koodia.")
            return

        aw = self._spin_aw.value()
        ah = self._spin_ah.value()
        self.segments, self.max_power = preprocess(self.commands)

        self.canvas.set_segments(self.segments, self.max_power, aw, ah)
        self._current_seg = 0

        self._spin_jump.setRange(0, len(self.segments))
        self._progress.setMaximum(max(1, len(self.segments)))
        self._progress.setValue(0)

        name = Path(path).name
        self._lbl_filename.setText(name)
        self.setWindowTitle(f"Laser-simulaattori – {name}")

        cmd_count = len(self.commands)
        seg_count = len(self.segments)
        self._update_seg_label()
        self.statusBar().showMessage(
            f"Ladattu: {cmd_count} käskyä → {seg_count} segmenttiä  |  "
            f"Max teho: {self.max_power:.0f}"
        )

    # ------------------------------------------------------------------
    # ANIMAATIO
    # ------------------------------------------------------------------

    def _play(self):
        """Käynnistää animaation. Jatkaa siitä mihin jäätiin."""
        if not self.segments:
            return
        if self._current_seg >= len(self.segments):
            self._reset_sim()
        self._playing = True
        # Käynnistä timer vain kerran – älä nollaa kumulatiivista aikaa
        self._elapsed.start()
        self._timer.start()
        self.statusBar().showMessage("Simulointi käynnissä…")

    def _pause(self):
        """Pysäyttää animaation ja tallentaa kuluneen ajan."""
        self._playing = False
        self._timer.stop()
        # Lisää tämän jakson aika kumulatiiviseen summaan
        if self._elapsed.isValid():
            self._elapsed_ms_accumulated += self._elapsed.elapsed()
        total_s = self._elapsed_ms_accumulated / 1000.0
        self.statusBar().showMessage(
            f"Tauolla – {self._current_seg}/{len(self.segments)} segmenttiä  "
            f"({total_s:.1f} s)"
        )

    def _reset_sim(self):
        """Nollaa simulaattorin alkuasentoon."""
        self._playing = False
        self._timer.stop()
        self._current_seg = 0
        self._elapsed_ms_accumulated = 0   # Nollaa kumulatiivinen aika
        self.canvas.advance_to(0)
        self._progress.setValue(0)
        self._spin_jump.setValue(0)
        self._lbl_gcline.setText("")
        self._update_seg_label()
        self.statusBar().showMessage("Nollattu.")

    def _jump_to_end(self):
        """Hyppää viimeiseen segmenttiin."""
        if self.segments:
            self._pause()
            self._goto_segment(len(self.segments))

    def _step_forward(self):
        """Yksi askel eteenpäin."""
        if self._current_seg < len(self.segments):
            self._goto_segment(self._current_seg + 1)

    def _step_backward(self):
        """Yksi askel taaksepäin."""
        if self._current_seg > 0:
            self._goto_segment(self._current_seg - 1)

    def _jump_to_segment(self):
        """Hyppää spin-boxin osoittamaan segmenttiin."""
        self._pause()
        self._goto_segment(self._spin_jump.value())

    def _goto_segment(self, idx: int):
        """Siirtyy tiettyyn segmentti-indeksiin."""
        idx = max(0, min(idx, len(self.segments)))
        self._current_seg = idx
        self.canvas.advance_to(idx)
        self._spin_jump.setValue(idx)
        self._progress.setValue(idx)
        self._update_seg_label()
        self._update_gcline_label()

    def _tick(self):
        """
        Timer-callback joka ajetaan ~60fps tahdissa.
        Etenee _steps_per_tick segmenttiä per tick.
        """
        if not self._playing or not self.segments:
            self._timer.stop()
            return

        next_seg = min(
            self._current_seg + self._steps_per_tick,
            len(self.segments)
        )
        self._current_seg = next_seg
        self.canvas.advance_to(next_seg)

        self._progress.setValue(next_seg)
        self._spin_jump.setValue(next_seg)
        self._update_seg_label()
        self._update_gcline_label()

        if next_seg >= len(self.segments):
            self._playing = False
            self._timer.stop()
            if self._elapsed.isValid():
                self._elapsed_ms_accumulated += self._elapsed.elapsed()
            total_s = self._elapsed_ms_accumulated / 1000.0
            self.statusBar().showMessage(
                f"Valmis – {len(self.segments)} segmenttiä  ({total_s:.1f} s)"
            )

    # ------------------------------------------------------------------
    # UI-PÄIVITYKSET
    # ------------------------------------------------------------------

    def _update_seg_label(self):
        n = len(self.segments)
        self._lbl_seg.setText(f"{self._current_seg} / {n} segmenttiä")
        # Päivitä myös teho-label
        if self.segments and 0 < self._current_seg <= len(self.segments):
            seg = self.segments[self._current_seg - 1]
            pct = seg.power / max(1, self.max_power) * 100
            self._lbl_power.setText(
                f"Teho: {seg.power:.0f}  ({pct:.0f}%)"
            )
        else:
            self._lbl_power.setText("Teho: 0")

    def _update_gcline_label(self):
        """Näyttää nykyisen G-koodin tekstimuodossa."""
        if not self.segments or self._current_seg == 0:
            self._lbl_gcline.setText("")
            return
        seg = self.segments[self._current_seg - 1]
        idx = seg.cmd_idx
        if 0 <= idx < len(self.commands):
            cmd = self.commands[idx]
            self._lbl_gcline.setText(
                f"Rivi {cmd.line_num}: {cmd.raw}"
            )

    def _on_mouse_mm(self, xmm: float, ymm: float):
        """Päivittää koordinaatit statusbariin hiiren liikkuessa."""
        self._lbl_pos.setText(f"X: {xmm:.3f}  Y: {ymm:.3f} mm")

    def _on_speed_changed(self, v: int):
        self._steps_per_tick = v
        self._lbl_speed.setText(f"{v} segm./frame")

    def _on_burn_changed(self, v: int):
        """Päivittää polttovoimakkuuden ja piirtää kuvan uudelleen."""
        factor = v / 100.0
        self.canvas.burn_intensity = factor
        self._lbl_burn.setText(f"{factor:.2f}×")
        self.canvas.redraw_all()

    def _on_vis_changed(self, v: bool):
        """Päivittää siirtoreittien näkyvyyden ja piirtää uudelleen."""
        self.canvas.show_travel = self._chk_travel.isChecked()
        self.canvas.redraw_all()

    def _on_area_changed(self):
        self.canvas.area_w_mm = self._spin_aw.value()
        self.canvas.area_h_mm = self._spin_ah.value()
        self.canvas._fit_view()
        self.canvas.update()

    def _fit_view(self):
        self.canvas.fit_view()

    # ------------------------------------------------------------------
    # VIENTI
    # ------------------------------------------------------------------

    def _export_png(self):
        """Vie nykyinen polttokuva PNG-tiedostoon Output-kansioon."""
        if not self.canvas._burn_img:
            QMessageBox.information(self, "Ei kuvaa", "Ei vietävää kuvaa.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Vie kuva",
            str(self.output_dir / "simulaatio.png"),
            "PNG (*.png);;Kaikki (*)"
        )
        if path:
            self.canvas._burn_img.save(path)
            self.statusBar().showMessage(f"Tallennettu: {path}")

    # ------------------------------------------------------------------
    # NÄPPÄINOIKOPOLUT
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            if self._playing:
                self._pause()
            else:
                self._play()
        elif event.key() == Qt.Key_R:
            self._reset_sim()
        elif event.key() == Qt.Key_Right and not self._playing:
            self._step_forward()
        elif event.key() == Qt.Key_Left and not self._playing:
            self._step_backward()
        elif event.key() == Qt.Key_End:
            self._jump_to_end()
        else:
            super().keyPressEvent(event)


    def closeEvent(self, event):
        """
        Käsittelee ikkunan sulkemisen.

        Pysäyttää animaatiotimerin ja kutsuu QApplication.quit()
        eksplisiittisesti jotta event loop varmasti pysähtyy.
        """
        self._playing = False
        self._timer.stop()
        event.accept()
        QApplication.quit()


# ===========================================================================
# KÄYNNISTYS
# ===========================================================================

def main():
    """Käynnistää Qt-sovelluksen. Hyväksyy valinnaisen G-kooditiedoston."""
    app = QApplication(sys.argv)
    app.setApplicationName("LaserSimulaattori")
    gcode_path = sys.argv[1] if len(sys.argv) > 1 else None
    win = SimulatorWindow(gcode_path)
    win.show()
    app.exec_()
    # Pakota prosessi sammumaan.
    os._exit(0)


if __name__ == "__main__":
    main()
