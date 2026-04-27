"""
main_app.py
===========
Laser-piirturin / CNC-kaivertimen ohjaussovellus.

Rakenne:
  ProcessingThread  – Taustasäie, jotta UI ei jäädy prosessoinnin aikana.
  ImageViewer       – QLabel-pohjainen kuvakatselukomponentti.
  ParamPanel        – Metodikohtaiset parametrisäädöt.
  LaserPlotterApp   – Pääikkuna, yhdistää kaikki osat.

Riippuvuudet:
  PyQt5, opencv-python, numpy  (ks. README.md)
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget, QTextEdit,
    QSplitter, QScrollArea, QMessageBox, QProgressBar,
    QGridLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from image_processor import (
    load_image, crop_image, scale_to_area, remove_background,
    method_edge_detection,
    method_hatch_shading,
    method_raster_spacing,
    method_raster_size,
    method_sine_wave,
    method_depth_map,
    GCodeGenerator,
)


# ===========================================================================
# APUFUNKTIOT
# ===========================================================================

def numpy_to_pixmap(img: np.ndarray,
                    max_w: int = 700,
                    max_h: int = 550) -> QPixmap:
    """
    Muuntaa numpy-kuvan (BGR tai harmaasävy) QPixmapiksi näyttöä varten.

    KORJATTU BUG: Alkuperäinen toteutus käytti .data.tobytes() NumPy-arrayn
    .data-attribuutille ilman yhtenäisyystarkistusta. Jos array ei ole
    C-yhtenäinen muistissa (esim. slice-operaation jälkeen), QImage sai
    virheellisen puskurin ja kuva näkyi vääristettynä tai kaatoi sovelluksen.
    Nyt np.ascontiguousarray() varmistaa yhtenäisen muistilohkon.

    Args:
        img:   NumPy uint8-kuva. BGR-kuva muunnetaan RGB:ksi Qt:lle.
        max_w: Suurin sallittu leveys pikseleinä (kuvasuhde säilyy).
        max_h: Suurin sallittu korkeus pikseleinä.

    Returns:
        Skaalattu QPixmap, tai tyhjä QPixmap jos img on None.
    """
    if img is None:
        return QPixmap()

    # Varmista C-yhtenäinen muisti ennen QImage-muunnosta
    img = np.ascontiguousarray(img)

    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)
    if pixmap.width() > max_w or pixmap.height() > max_h:
        pixmap = pixmap.scaled(
            max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return pixmap


# ===========================================================================
# TAUSTASÄIE PROSESSOINTIIN
# ===========================================================================

class ProcessingThread(QThread):
    """
    Ajaa kuvanprosessointifunktion erillisessä säikeessä, jotta
    Qt:n käyttöliittymä pysyy responsiivisena prosessoinnin aikana.

    Signaalit:
        result_ready(object)  – Lähetetään onnistuneen prosessoinnin jälkeen.
        error_occurred(str)   – Lähetetään jos poikkeus tapahtui.
    """

    result_ready = pyqtSignal(object)   # object, koska np.ndarray ei ole Qt-tyyppi
    error_occurred = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        """
        Args:
            func:     Kutsuttava funktio (image_processor-moduulista).
            *args:    Funktion positioargumentit.
            **kwargs: Funktion avainsanaargumentit.
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Säikeen suoritusmetodi – kutsuu func(*args, **kwargs)."""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ===========================================================================
# KUVAKATSELUKOMPONENTTI
# ===========================================================================

class ImageViewer(QLabel):
    """
    Kuvanäyttökomponentti, joka skaalaa kuvan automaattisesti
    komponentin kokoon.

    Tallentaa alkuperäisen pixmapin (_source_pixmap) erillään näytettävästä,
    jotta uudelleenskaalaus ei heikennä kuvanlaatua.
    """

    def __init__(self, placeholder_text: str = ""):
        super().__init__()
        self._source_pixmap = None
        self.placeholder_text = placeholder_text

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(280, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 1px solid #444;
                border-radius: 4px;
                color: #666;
                font-size: 13px;
            }
        """)
        self.setText(f"[ {placeholder_text} ]")

    def set_image(self, img: np.ndarray):
        """
        Asettaa näytettävän kuvan.

        KORJATTU BUG: Alkuperäinen toteutus kutsui numpy_to_pixmap()
        komponentin silloisen koon mukaan. Komponentti ei välttämättä ole
        oikeassa koossaan vielä layoutin alustuksessa, joten kuva näkyi
        liian pienenä tai se puuttui kokonaan. Nyt tallennetaan
        täysikokoinen lähde ja skaalataan resizeEvent:ssä.

        Args:
            img: NumPy-kuva tai None (palauttaa placeholdertekstin).
        """
        if img is None:
            self._source_pixmap = None
            self.setText(f"[ {self.placeholder_text} ]")
            return

        # Luo täysikokoinen lähde-pixmap
        img = np.ascontiguousarray(img)
        if len(img.shape) == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

        self._source_pixmap = QPixmap.fromImage(qimg)
        self._refresh_display()

    def _refresh_display(self):
        """Skaalaa tallennetun pixmapin komponentin nykyiseen kokoon."""
        if self._source_pixmap is None:
            return
        scaled = self._source_pixmap.scaled(
            max(1, self.width() - 4),
            max(1, self.height() - 4),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        """Päivitetään kuva aina kun komponentin koko muuttuu."""
        super().resizeEvent(event)
        self._refresh_display()


# ===========================================================================
# METODIKOHTAINEN PARAMETRIPANEELI
# ===========================================================================

class ParamPanel(QWidget):
    """
    Dynaaminen parametripaneeli: näyttää kulloisenkin metodin omat säädöt
    sekä aina näkyvissä olevat yhteiset asetukset.

    Rakenne:
      - Yhteiset asetukset: pikselikoko, dwell-aika.
      - Metodin parametrit: rakennetaan METHOD_PARAMS-taulukosta.

    Signaali:
        params_changed – Lähetetään aina kun jokin arvo muuttuu.
    """

    params_changed = pyqtSignal()

    # Metodikohtaiset parametrimäärittelyt.
    # Rakenne: {avain: (otsikko, oletus, minimi, maksimi, askel, on_double)}
    METHOD_PARAMS = {
        0: {  # Reunantunnistus
            "low_thresh":  ("Alempi kynnys",          50,  1,   254, 1,    False),
            "high_thresh": ("Ylempi kynnys",          150, 2,   255, 1,    False),
            "min_line":    ("Min. viivan pituus (px)", 10,  0,   500, 1,    False),
        },
        1: {  # Hatch-varjostus
            "levels":       ("Sävymäärä (posteris.)",  8,  2,  32,  1,    False),
            "line_spacing": ("Maksimiviivaväli (px)",  6,  1,  50,  1,    False),
            "angle":        ("Viivojen kulma (°)",     0,  0, 180,  1.0,  True),
        },
        2: {  # Rasterointi – etäisyys
            "grid_size":   ("Solukoko (px)",     10, 2, 50, 1,    False),
            "dot_radius":  ("Pisteen säde (px)",  2, 1, 20, 1,    False),
        },
        3: {  # Rasterointi – koko
            "grid_size":  ("Solukoko (px)",       10,  2,    50, 1,    False),
            "max_radius": ("Maks. säde-suhde",  0.45, 0.10, 0.49, 0.01, True),
        },
        4: {  # Siniaalto
            "line_gap":  ("Riviväli (px)",       12,  4,  80, 1,    False),
            "amp_scale": ("Amplitudikerroin",   1.0, 0.1, 3.0, 0.1, True),
        },
        5: {  # Syvyyskartta
            "levels":  ("Syvyystasoja",        16, 2,   64, 1,    False),
            "z_depth": ("Maks. syvyys (mm)",  3.0, 0.1, 20.0, 0.5, True),
        },
    }

    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self.widgets = {}
        self._build(0)

    def _build(self, method_idx: int):
        """
        Rakentaa parametripaneelin valitulle metodille.
        Tyhjentää ensin aiemmat widgetit.
        """
        # Tyhjennä vanha sisältö
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.widgets = {}

        # --- Yhteiset asetukset (näkyvissä aina) ---
        common_grp = QGroupBox("Yhteiset asetukset")
        cgl = QGridLayout(common_grp)

        cgl.addWidget(QLabel("Pikselikoko (mm/px):"), 0, 0)
        self.widgets["pixel_size"] = self._make_double(0.1, 0.01, 1.0, 0.01)
        self.widgets["pixel_size"].setToolTip(
            "Yhden pikselin fyysinen koko millimetreinä.\n"
            "Vaikuttaa G-koodin X/Y-koordinaatteihin."
        )
        cgl.addWidget(self.widgets["pixel_size"], 0, 1)

        cgl.addWidget(QLabel("Dwell-aika ms (metodi 4):"), 1, 0)
        self.widgets["dwell_ms"] = self._make_int(50, 0, 5000, 10)
        self.widgets["dwell_ms"].setToolTip(
            "Viipymäaika (G4 P<ms>) per pistepoltto.\n"
            "Pidempi aika = tummempi poltto."
        )
        cgl.addWidget(self.widgets["dwell_ms"], 1, 1)

        self._layout.addWidget(common_grp)

        # --- Metodin omat parametrit ---
        specs = self.METHOD_PARAMS.get(method_idx, {})
        if specs:
            method_grp = QGroupBox("Metodin parametrit")
            mgl = QGridLayout(method_grp)
            for row, (key, (label, default, lo, hi, step, is_double)) \
                    in enumerate(specs.items()):
                mgl.addWidget(QLabel(f"{label}:"), row, 0)
                w = (self._make_double(default, lo, hi, step)
                     if is_double else self._make_int(default, lo, hi, step))
                self.widgets[key] = w
                mgl.addWidget(w, row, 1)
            self._layout.addWidget(method_grp)

        self._layout.addStretch()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Apumetodit widgettien luontiin
    # ------------------------------------------------------------------

    def _make_int(self, default, lo, hi, step) -> QSpinBox:
        """Luo ja palauttaa QSpinBox annetuilla arvoilla."""
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(default)
        return w

    def _make_double(self, default, lo, hi, step) -> QDoubleSpinBox:
        """Luo ja palauttaa QDoubleSpinBox annetuilla arvoilla."""
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(2)
        w.setValue(default)
        return w

    def _connect_signals(self):
        """Kytkee kaikkien widgetien muutossignaalit params_changed-signaaliin."""
        for w in self.widgets.values():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self.params_changed)
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.params_changed)

    # ------------------------------------------------------------------
    # Julkinen API
    # ------------------------------------------------------------------

    def update_for_method(self, method_idx: int):
        """Rakentaa paneelin uudelleen valitulle metodille."""
        self._build(method_idx)

    def get_params(self) -> dict:
        """
        Palauttaa kaikki nykyiset parametriarvot sanakirjana.

        Returns:
            {avain: int/float/str}
        """
        result = {}
        for key, w in self.widgets.items():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                result[key] = w.value()
            elif isinstance(w, QComboBox):
                result[key] = w.currentText()
        return result


# ===========================================================================
# PÄÄIKKUNA
# ===========================================================================

class LaserPlotterApp(QMainWindow):
    """
    Sovelluksen pääikkuna.

    Layout:
      Vasen paneeli (kiinteä leveys):
        - Kuvan lataus (tiedosto / kamera / henkilökuvamoodi)
        - Toimilaitevalinta ja G-koodin perusparametrit
        - Prosessointimetodivalinta
        - Rajauskoordinaatit
        - Metodikohtaiset parametrit (vierityspaneelissa)
        - Prosessoi / Generoi G-koodi -painikkeet

      Oikea paneeli (välilehdet):
        - Kuvat: alkuperäinen ja tulos rinnakkain
        - G-koodi: teksti, tallennus, kopiointi
        - Alue & Laite: piirtoalueen mitat ja tallennushakemisto
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laser-piirturi / CNC-kaiverrin – Kuvamuokkain")
        self.resize(1400, 860)

        # --- Sovellustila ---
        self.orig_img = None      # Alkuperäinen ladattu BGR-kuva (muuttumaton)
        self.work_img = None      # Skaalattu työkuva (muuttuu rajauksen/taustan mukaan)
        self.result_img = None    # Prosessoitu kuva (binääri tai harmaasävy)
        self.gcode_text = ""      # Viimeksi generoitu G-koodi merkkijonona
        self.proc_thread = None   # Viittaus käynnissä olevaan ProcessingThread:iin

        self.output_dir = Path.home() / "laser_output"
        self.output_dir.mkdir(exist_ok=True)

        self._apply_dark_theme()
        self._build_ui()

        # Kytke metodivalinta parametripaneeliin (molemmat nyt luotu)
        self.method_combo.currentIndexChanged.connect(
            self.param_panel.update_for_method
        )

        self.statusBar().showMessage(
            "Valmis – lataa kuva tiedostosta tai kameralta"
        )

    # ------------------------------------------------------------------
    # TYYLI
    # ------------------------------------------------------------------

    def _apply_dark_theme(self):
        """Asettaa tumman teeman Qt-tyyleinä."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #16213e;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #0f3460;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: bold;
                color: #4fc3f7;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #4fc3f7;
            }
            QPushButton {
                background-color: #0f3460;
                border: 1px solid #1565c0;
                border-radius: 5px;
                padding: 6px 14px;
                color: #e0e0e0;
                min-height: 28px;
            }
            QPushButton:hover  { background-color: #1565c0; }
            QPushButton:pressed { background-color: #0d47a1; }
            QPushButton#btn_process {
                background-color: #1b5e20;
                border-color: #2e7d32;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton#btn_process:hover { background-color: #2e7d32; }
            QPushButton#btn_gcode {
                background-color: #4a148c;
                border-color: #7b1fa2;
            }
            QPushButton#btn_gcode:hover { background-color: #7b1fa2; }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #0f3460;
                border: 1px solid #1565c0;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0e0;
                min-height: 24px;
            }
            QTextEdit {
                background-color: #0a0a1a;
                border: 1px solid #333;
                color: #a5d6a7;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
            QTabWidget::pane  { border: 1px solid #0f3460; }
            QTabBar::tab {
                background: #0f3460;
                padding: 6px 18px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                color: #aaa;
            }
            QTabBar::tab:selected { background: #1565c0; color: #fff; }
            QProgressBar {
                border: 1px solid #0f3460;
                border-radius: 3px;
                background: #0a0a1a;
                max-height: 8px;
            }
            QProgressBar::chunk { background: #4fc3f7; }
            QScrollArea { border: none; }
        """)

    # ------------------------------------------------------------------
    # UI-RAKENNE
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Rakentaa pääikkunan juuritason layoutin."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel(), stretch=1)

    def _build_left_panel(self) -> QWidget:
        """Vasen ohjauspaneeli."""
        panel = QWidget()
        panel.setFixedWidth(316)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(self._build_image_group())
        layout.addWidget(self._build_device_group())
        layout.addWidget(self._build_method_group())
        layout.addWidget(self._build_crop_group())

        # Parametripaneeli vierityslaatikossa
        self.param_panel = ParamPanel()
        scroll = QScrollArea()
        scroll.setWidget(self.param_panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(235)
        layout.addWidget(scroll)

        # Toimintopainikkeet
        btn_process = QPushButton("▶  PROSESSOI KUVA")
        btn_process.setObjectName("btn_process")
        btn_process.setToolTip("Prosessoi kuva valitulla metodilla")
        btn_process.clicked.connect(self.run_processing)
        layout.addWidget(btn_process)

        btn_gcode = QPushButton("⚙  Generoi G-koodi")
        btn_gcode.setObjectName("btn_gcode")
        btn_gcode.setToolTip("Muunna tulos G-koodiksi valitulle laitteelle")
        btn_gcode.clicked.connect(self.generate_gcode)
        layout.addWidget(btn_gcode)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        return panel

    def _build_right_panel(self) -> QTabWidget:
        """Oikea välilehtipaneeli."""
        tabs = QTabWidget()

        # Välilehti 1: Kuvat
        img_tab = QWidget()
        il = QVBoxLayout(img_tab)
        splitter = QSplitter(Qt.Horizontal)

        orig_box = QGroupBox("Alkuperäinen / Työkuva")
        obl = QVBoxLayout(orig_box)
        self.view_orig = ImageViewer("Alkuperäinen")
        obl.addWidget(self.view_orig)
        splitter.addWidget(orig_box)

        res_box = QGroupBox("Prosessoitu kuva")
        rbl = QVBoxLayout(res_box)
        self.view_result = ImageViewer("Tulos")
        rbl.addWidget(self.view_result)
        splitter.addWidget(res_box)

        il.addWidget(splitter)
        tabs.addTab(img_tab, "📷  Kuvat")

        # Välilehti 2: G-koodi
        gcode_tab = QWidget()
        gcl = QVBoxLayout(gcode_tab)
        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾  Tallenna .gcode")
        btn_save.clicked.connect(self.save_gcode)
        btn_copy = QPushButton("📋  Kopioi")
        btn_copy.clicked.connect(self.copy_gcode)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        gcl.addLayout(btn_row)
        self.gcode_view = QTextEdit()
        self.gcode_view.setReadOnly(True)
        self.gcode_view.setPlaceholderText(
            "; G-koodi ilmestyy tähän prosessoinnin jälkeen.\n"
            "; Valitse metodi → Prosessoi → Generoi G-koodi."
        )
        gcl.addWidget(self.gcode_view)
        tabs.addTab(gcode_tab, "⚙  G-koodi")

        # Välilehti 3: Alue & Laite
        tabs.addTab(self._build_area_tab(), "📐  Alue & Laite")
        return tabs

    def _build_image_group(self) -> QGroupBox:
        """Kuvan lataus -ryhmä."""
        grp = QGroupBox("Kuvan lataus")
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        btn_file = QPushButton("📂  Tiedostosta")
        btn_file.clicked.connect(self.load_from_file)
        btn_cam = QPushButton("📷  Kameralta")
        btn_cam.clicked.connect(self.load_from_camera)
        row.addWidget(btn_file)
        row.addWidget(btn_cam)
        layout.addLayout(row)

        self.chk_portrait = QCheckBox("Henkilökuva – poista tausta")
        self.chk_portrait.setToolTip(
            "GrabCut-algoritmi poistaa taustan.\n"
            "Toimii parhaiten kun kohde on kuvan keskialueella."
        )
        self.chk_portrait.toggled.connect(self._on_portrait_toggled)
        layout.addWidget(self.chk_portrait)
        return grp

    def _build_device_group(self) -> QGroupBox:
        """Toimilaite-ryhmä."""
        grp = QGroupBox("Toimilaite")
        gl = QGridLayout(grp)

        gl.addWidget(QLabel("Laite:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems([
            "Laser (polttaminen)",
            "X-Y-piirturi (kynä)",
            "CNC-kaiverrin",
        ])
        gl.addWidget(self.device_combo, 0, 1)

        gl.addWidget(QLabel("Syöttönopeus (mm/min):"), 1, 0)
        self.spin_feed = QSpinBox()
        self.spin_feed.setRange(10, 20000)
        self.spin_feed.setSingleStep(100)
        self.spin_feed.setValue(1000)
        gl.addWidget(self.spin_feed, 1, 1)

        gl.addWidget(QLabel("Laserin max-teho (S):"), 2, 0)
        self.spin_power = QSpinBox()
        self.spin_power.setRange(1, 10000)
        self.spin_power.setSingleStep(50)
        self.spin_power.setValue(1000)
        self.spin_power.setToolTip("GRBL: 255 tai 1000. Marlin: 255.")
        gl.addWidget(self.spin_power, 2, 1)

        gl.addWidget(QLabel("Z turvakorkeus (mm):"), 3, 0)
        self.spin_z_up = QDoubleSpinBox()
        self.spin_z_up.setRange(0.0, 50.0)
        self.spin_z_up.setSingleStep(0.5)
        self.spin_z_up.setValue(5.0)
        gl.addWidget(self.spin_z_up, 3, 1)

        return grp

    def _build_method_group(self) -> QGroupBox:
        """Prosessointimetodi-ryhmä."""
        grp = QGroupBox("Prosessointimetodi")
        layout = QVBoxLayout(grp)
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "1 – Reunantunnistus (Canny)",
            "2 – Varjostettu viivapiirros (Hatch)",
            "3 – Rasterointi, etäisyys vaihtelee",
            "4 – Rasterointi, koko vaihtelee",
            "5 – Siniaaltorasterointi",
            "6 – Syvyyskartta (Z-kaiverrus)",
        ])
        layout.addWidget(self.method_combo)
        return grp

    def _build_crop_group(self) -> QGroupBox:
        """Rajauskoordinaattien syöttöryhmä."""
        grp = QGroupBox("Rajaus (pikseleinä)")
        gl = QGridLayout(grp)

        self.spin_x1 = QSpinBox(); self.spin_x1.setRange(0, 99999)
        self.spin_y1 = QSpinBox(); self.spin_y1.setRange(0, 99999)
        self.spin_x2 = QSpinBox(); self.spin_x2.setRange(0, 99999)
        self.spin_y2 = QSpinBox(); self.spin_y2.setRange(0, 99999)

        gl.addWidget(QLabel("X1:"), 0, 0); gl.addWidget(self.spin_x1, 0, 1)
        gl.addWidget(QLabel("Y1:"), 0, 2); gl.addWidget(self.spin_y1, 0, 3)
        gl.addWidget(QLabel("X2:"), 1, 0); gl.addWidget(self.spin_x2, 1, 1)
        gl.addWidget(QLabel("Y2:"), 1, 2); gl.addWidget(self.spin_y2, 1, 3)

        btn_reset = QPushButton("Nollaa rajaus")
        btn_reset.clicked.connect(self._reset_crop)
        gl.addWidget(btn_reset, 2, 0, 1, 4)
        return grp

    def _build_area_tab(self) -> QWidget:
        """Piirtoalueen mitat ja tallennushakemisto -välilehti."""
        w = QWidget()
        layout = QVBoxLayout(w)

        area_grp = QGroupBox("Piirtoalueen mitat")
        agl = QGridLayout(area_grp)

        agl.addWidget(QLabel("Leveys (mm):"), 0, 0)
        self.spin_area_w = QDoubleSpinBox()
        self.spin_area_w.setRange(10, 2000)
        self.spin_area_w.setValue(200)
        agl.addWidget(self.spin_area_w, 0, 1)

        agl.addWidget(QLabel("Korkeus (mm):"), 1, 0)
        self.spin_area_h = QDoubleSpinBox()
        self.spin_area_h.setRange(10, 2000)
        self.spin_area_h.setValue(150)
        agl.addWidget(self.spin_area_h, 1, 1)

        agl.addWidget(QLabel("Resoluutio (DPI):"), 2, 0)
        self.spin_dpi = QDoubleSpinBox()
        self.spin_dpi.setRange(10, 1200)
        self.spin_dpi.setValue(96)
        agl.addWidget(self.spin_dpi, 2, 1)

        layout.addWidget(area_grp)

        dir_grp = QGroupBox("Tallennushakemisto")
        dgl = QHBoxLayout(dir_grp)
        self.lbl_outdir = QLabel(str(self.output_dir))
        self.lbl_outdir.setWordWrap(True)
        btn_dir = QPushButton("Vaihda…")
        btn_dir.clicked.connect(self._choose_outdir)
        dgl.addWidget(self.lbl_outdir, stretch=1)
        dgl.addWidget(btn_dir)
        layout.addWidget(dir_grp)

        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # TAPAHTUMANKÄSITTELIJÄT
    # ------------------------------------------------------------------

    def _on_portrait_toggled(self, checked: bool):
        """
        Käsittelee henkilökuva-valintaruudun tilan muutoksen.

        Ruksattu:     Poistetaan tausta GrabCut-algoritmilla.
        Pois päältä:  Palautetaan alkuperäinen skaalattu työkuva.
        """
        if self.orig_img is None:
            return

        if checked:
            self.statusBar().showMessage(
                "Poistetaan taustaa GrabCut-algoritmilla…"
            )
            QApplication.processEvents()  # Päivitä statusbari ennen hidasta operaatiota
            try:
                result, _mask = remove_background(self.work_img)
                self.work_img = result
                self.view_orig.set_image(result)
                self.statusBar().showMessage("Tausta poistettu.")
            except Exception as exc:
                QMessageBox.warning(self, "Taustan poisto epäonnistui", str(exc))
                # Palauta ruksi pois päältä ilman toistuvia signaaleja
                self.chk_portrait.blockSignals(True)
                self.chk_portrait.setChecked(False)
                self.chk_portrait.blockSignals(False)
        else:
            # Palauta alkuperäinen skaalattu versio
            self.work_img = scale_to_area(
                self.orig_img,
                self.spin_area_w.value(),
                self.spin_area_h.value(),
                self.spin_dpi.value(),
            )
            self.view_orig.set_image(self.work_img)
            self.statusBar().showMessage("Alkuperäinen kuva palautettu.")

    def _reset_crop(self):
        """Asettaa rajauskoordinaatit työkuvan täyteen kokoon."""
        ref = self.work_img if self.work_img is not None else self.orig_img
        if ref is None:
            return
        h, w = ref.shape[:2]
        self.spin_x1.setValue(0); self.spin_y1.setValue(0)
        self.spin_x2.setValue(w); self.spin_y2.setValue(h)

    def _choose_outdir(self):
        """Avaa hakemistonvalintaikkunan tallennushakemiston vaihtamiseksi."""
        path = QFileDialog.getExistingDirectory(
            self, "Valitse tallennushakemisto", str(self.output_dir)
        )
        if path:
            self.output_dir = Path(path)
            self.lbl_outdir.setText(str(self.output_dir))

    # ------------------------------------------------------------------
    # KUVAN LATAUS
    # ------------------------------------------------------------------

    def load_from_file(self):
        """Avaa tiedostonvalintaikkunan ja lataa valitun kuvan."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Avaa kuva", str(Path.home()),
            "Kuvat (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if path:
            self._load_and_prepare(path)

    def load_from_camera(self):
        """
        Ottaa kuvan ensimmäisellä löydetyllä USB-kameralla ja tallentaa sen.

        KORJATTU BUG: Alkuperäinen koodi ei vapauttanut kameraa
        virhetilanteessa. Nyt cap.release() on try/finally-lohkossa, jolloin
        kamera vapautetaan aina – myös jos cap.read() heittää poikkeuksen.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(
                self, "Kamera",
                "Kameraa ei löydy tai se on toisen sovelluksen käytössä."
            )
            return

        try:
            ret, frame = cap.read()
        finally:
            cap.release()   # Vapautetaan aina

        if not ret:
            QMessageBox.warning(self, "Kamera", "Kuvan ottaminen epäonnistui.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"camera_{ts}.jpg"
        cv2.imwrite(str(save_path), frame)
        self._load_and_prepare(str(save_path))
        self.statusBar().showMessage(f"Tallennettu: {save_path.name}")

    def _load_and_prepare(self, path: str):
        """
        Lataa kuvatiedoston, skaalaa piirtoalueelle ja päivittää UI:n.

        KORJATTU BUG: Alkuperäinen koodi ylikirjoitti orig_img:n skaalatulla
        versiolla. Tämä esti rajauksen nollauksen ja taustanpoiston
        peruutuksen toimimisen oikein. Nyt:
          orig_img = muuttumaton alkuperäinen (käytetään nollaukseen)
          work_img = skaalattu kopio (käytetään prosessointiin)

        Args:
            path: Ladattavan kuvatiedoston polku.
        """
        try:
            self.orig_img = load_image(path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Latausvirhe", str(exc))
            return

        # Skaalaa työkuva piirtoalueen kokoon kuvasuhde säilyttäen
        self.work_img = scale_to_area(
            self.orig_img,
            self.spin_area_w.value(),
            self.spin_area_h.value(),
            self.spin_dpi.value(),
        )

        # Nollaa rajauskoordinaatit työkuvan koon mukaan
        h, w = self.work_img.shape[:2]
        self.spin_x1.setValue(0); self.spin_y1.setValue(0)
        self.spin_x2.setValue(w); self.spin_y2.setValue(h)

        # Nollaa sovellustila
        self.result_img = None
        self.chk_portrait.blockSignals(True)
        self.chk_portrait.setChecked(False)
        self.chk_portrait.blockSignals(False)

        self.view_orig.set_image(self.work_img)
        self.view_result.set_image(None)

        oh, ow = self.orig_img.shape[:2]
        self.statusBar().showMessage(
            f"Ladattu: {Path(path).name}  "
            f"(alkup. {ow}×{oh} px  →  työ {w}×{h} px)"
        )

    # ------------------------------------------------------------------
    # PROSESSOINTI
    # ------------------------------------------------------------------

    def _get_cropped_work_img(self) -> np.ndarray | None:
        """
        Palauttaa rajatun työkuvan spin-boxien koordinaattien mukaan.

        crop_image() hoitaa reunatarkistukset, joten koordinaatit
        voivat ylittää kuvan reunat turvallisesti.
        """
        if self.work_img is None:
            return None
        return crop_image(
            self.work_img,
            self.spin_x1.value(), self.spin_y1.value(),
            self.spin_x2.value(), self.spin_y2.value(),
        )

    def run_processing(self):
        """
        Käynnistää kuvanprosessoinnin taustasäikeessä.

        KORJATTU BUG: Alkuperäinen koodi ei tarkistanut onko edellinen
        prosessointi vielä kesken. Uusi kutsu olisi luonut toisen säikeen
        ja ylikirjoittanut self.proc_thread-viitteen, jolloin vanha säie
        olisi jäänyt "haamuksi" ilman siivousmahdollisuutta.
        Nyt tarkistetaan isRunning() ennen uuden käynnistystä.
        """
        if self.work_img is None:
            QMessageBox.information(
                self, "Ei kuvaa", "Lataa ensin kuva ennen prosessointia."
            )
            return

        if self.proc_thread is not None and self.proc_thread.isRunning():
            self.statusBar().showMessage("Prosessointi jo käynnissä…")
            return

        img = self._get_cropped_work_img()
        method = self.method_combo.currentIndex()
        params = self.param_panel.get_params()

        def do_process():
            """
            Prosessointifunktio, joka ajetaan taustasäikeessä.
            Valitsee image_processor-moduulin oikean funktion.
            """
            if method == 0:
                return method_edge_detection(
                    img,
                    low_thresh=int(params.get("low_thresh", 50)),
                    high_thresh=int(params.get("high_thresh", 150)),
                    min_line_length=int(params.get("min_line", 10)),
                )
            elif method == 1:
                return method_hatch_shading(
                    img,
                    levels=int(params.get("levels", 8)),
                    line_spacing=int(params.get("line_spacing", 6)),
                    angle_deg=float(params.get("angle", 0.0)),
                )
            elif method == 2:
                return method_raster_spacing(
                    img,
                    dot_radius=int(params.get("dot_radius", 2)),
                    grid_size=int(params.get("grid_size", 10)),
                )
            elif method == 3:
                return method_raster_size(
                    img,
                    grid_size=int(params.get("grid_size", 10)),
                    max_radius_ratio=float(params.get("max_radius", 0.45)),
                )
            elif method == 4:
                return method_sine_wave(
                    img,
                    line_gap=int(params.get("line_gap", 12)),
                    amplitude_scale=float(params.get("amp_scale", 1.0)),
                )
            elif method == 5:
                return method_depth_map(
                    img,
                    levels=int(params.get("levels", 16)),
                )
            raise ValueError(f"Tuntematon metodi: {method}")

        # Käynnistä taustasäie
        self.proc_thread = ProcessingThread(do_process)
        self.proc_thread.result_ready.connect(self._on_result_ready)
        self.proc_thread.error_occurred.connect(self._on_proc_error)
        self.proc_thread.start()

        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate-animaatio
        self.statusBar().showMessage("Prosessoidaan…")

    def _on_result_ready(self, result: np.ndarray):
        """
        Kutsutaan taustasäikeestä onnistuneen prosessoinnin jälkeen.
        Päivittää näytön ja tallentaa tuloskuvan automaattisesti.
        """
        self.result_img = result
        self.progress_bar.hide()
        self.view_result.set_image(result)

        # Automaattitallennus
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_label = (
            self.method_combo.currentText().split("–")[0].strip()
        )
        out_path = self.output_dir / f"result_{method_label}_{ts}.png"
        cv2.imwrite(str(out_path), result)

        self.statusBar().showMessage(
            f"Prosessointi valmis → {out_path.name}  "
            f"({result.shape[1]}×{result.shape[0]} px)"
        )

    def _on_proc_error(self, message: str):
        """Kutsutaan taustasäikeestä jos prosessointi heitti poikkeuksen."""
        self.progress_bar.hide()
        QMessageBox.critical(self, "Prosessointivirhe", message)
        self.statusBar().showMessage("Prosessointi epäonnistui.")

    # ------------------------------------------------------------------
    # G-KOODI
    # ------------------------------------------------------------------

    def generate_gcode(self):
        """
        Generoi G-koodin prosessoidusta kuvasta ja näyttää sen
        G-koodi-välilehdellä.

        Generointimetodi riippuu prosessointimetodista:
          Metodi 5 (syvyyskartta)    → from_grayscale_image (Z-tasot)
          Metodi 3 (koko vaihtelee)  → from_dot_image (G4 dwell per piste)
          Muut                       → from_binary_image (jatkuva viiva)
        """
        if self.result_img is None:
            QMessageBox.information(
                self, "Ei tulosta",
                "Prosessoi kuva ensin ennen G-koodin generointia."
            )
            return

        # Selvitä toimilaite tekstistä
        device_text = self.device_combo.currentText()
        if "Laser" in device_text:
            device = "laser"
        elif "piirturi" in device_text:
            device = "plotter"
        else:
            device = "cnc"

        params = self.param_panel.get_params()
        gen = GCodeGenerator(
            device=device,
            feed_rate=int(self.spin_feed.value()),
            laser_power_max=int(self.spin_power.value()),
            z_up=float(self.spin_z_up.value()),
            pixel_size_mm=float(params.get("pixel_size", 0.1)),
        )

        method = self.method_combo.currentIndex()
        dwell_ms = int(params.get("dwell_ms", 50))

        try:
            if method == 5:
                z_depth = float(params.get("z_depth", 3.0))
                gcode = gen.from_grayscale_image(
                    self.result_img, z_max_depth=z_depth
                )
            elif method == 3:
                gcode = gen.from_dot_image(
                    self.result_img, dwell_ms=dwell_ms
                )
            else:
                gcode = gen.from_binary_image(self.result_img)

        except Exception as exc:
            QMessageBox.critical(self, "G-koodi-virhe", str(exc))
            return

        self.gcode_text = gcode
        self.gcode_view.setPlainText(gcode)
        line_count = gcode.count("\n") + 1
        self.statusBar().showMessage(
            f"G-koodi generoitu – {line_count} riviä"
        )

    def save_gcode(self):
        """Tallentaa generoidun G-koodin tiedostoon."""
        if not self.gcode_text:
            QMessageBox.information(self, "Ei G-koodia", "Generoi G-koodi ensin.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Tallenna G-koodi",
            str(self.output_dir / "output.gcode"),
            "G-koodi (*.gcode *.nc *.txt);;Kaikki (*)",
        )
        if path:
            Path(path).write_text(self.gcode_text, encoding="utf-8")
            self.statusBar().showMessage(f"G-koodi tallennettu: {path}")

    def copy_gcode(self):
        """Kopioi G-koodin leikepöydälle."""
        if self.gcode_text:
            QApplication.clipboard().setText(self.gcode_text)
            self.statusBar().showMessage("G-koodi kopioitu leikepöydälle.")
        else:
            self.statusBar().showMessage("Ei G-koodia kopioitavaksi.")


# ===========================================================================
# KÄYNNISTYS
# ===========================================================================

def main():
    """Luo Qt-sovelluksen, pääikkunan ja käynnistää tapahtumasilmukan."""
    app = QApplication(sys.argv)
    app.setApplicationName("LaserPlotter")
    window = LaserPlotterApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
