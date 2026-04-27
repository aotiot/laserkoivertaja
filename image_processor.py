"""
image_processor.py
==================
Kuvanmuokkausmoottori laser-piirturille ja CNC-kaivertimelle.

Tukee kuutta eri prosessointimetodia:
  1. Reunantunnistus   – Canny-algoritmi + pienten viivojen suodatus
  2. Hatch-varjostus   – posterisoidut harmaasävyt viivatiheytenä
  3. Rasterointi/etäis – Floyd-Steinberg dithering, vakiokoko
  4. Rasterointi/koko  – pistekoko skaalautuu tummuuden mukaan
  5. Siniaalto         – amplitudi vaihtelee tummuuden mukaan
  6. Syvyyskartta      – harmaasävy muunnetaan Z-kaiverrusarvoksi

Moduuli ei sisällä Qt-riippuvuuksia – se on puhdas kuvanmuokkaus-
kirjasto, jota main_app.py kutsuu.
"""

import math
import cv2
import numpy as np


# ===========================================================================
# APUFUNKTIOT
# ===========================================================================

def load_image(path: str) -> np.ndarray:
    """
    Lataa kuvatiedosto BGR-väriavaruuteen (OpenCV:n oletusmuoto).

    Raises:
        FileNotFoundError: jos tiedostoa ei löydy tai se ei ole kuva.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Kuvatiedostoa ei löydy tai se ei aukea: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Muuntaa BGR-kuvan harmaasävyksi. Jos kuva on jo harmaasävy, palauttaa kopion.

    Kopio varmistaa, että paluuarvo on aina kirjoitettavissa
    (ei viite alkuperäiseen).
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def crop_image(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Rajaa kuvan annettuun suorakulmaan (pikselikoordinaatit).
    Koordinaatit rajataan automaattisesti kuvan rajoihin.
    """
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return img[y1:y2, x1:x2]


def scale_to_area(img: np.ndarray,
                  width_mm: float,
                  height_mm: float,
                  dpi: float = 96) -> np.ndarray:
    """
    Skaalaa kuvan piirtoalueen fyysistä kokoa vastaavaan pikseliresoluutioon.

    Kuvasuhde säilytetään: kuva mahtuu alueeseen leikkaamatta.

    Args:
        img:       Lähde BGR- tai harmaasävykuvana.
        width_mm:  Piirtoalueen leveys millimetreinä.
        height_mm: Piirtoalueen korkeus millimetreinä.
        dpi:       Tulostusresoluutio (pistettä tuumalla).
    """
    target_w = int(width_mm / 25.4 * dpi)
    target_h = int(height_mm / 25.4 * dpi)
    src_h, src_w = img.shape[:2]

    # Laske skaalauskerroin niin, että kuva mahtuu alueeseen
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def posterize(gray: np.ndarray, levels: int) -> np.ndarray:
    """
    Posterisoi harmaasävykuva pienentämällä sävymäärän.

    Esim. levels=4 → neljä tasoa: 0, 85, 170, 255.

    KORJATTU BUG: Alkuperäinen toteutus käytti uint8-aritmetiikkaa
    ( np.uint8 * int(step) ), joka ylivuotaa kun step > 1 ja arvo > 127.
    Nyt käytetään float32-välivaihetta ylivuodon estämiseksi.

    Args:
        gray:   Harmaasävykuva (uint8, 0–255).
        levels: Sävytasojen määrä (vähintään 2).
    """
    levels = max(2, levels)
    step = 255.0 / (levels - 1)
    # float32-laskenta estää ylivuodon
    quantized = np.round(gray.astype(np.float32) / step) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)


def remove_background(img: np.ndarray) -> tuple:
    """
    Poistaa kuvan taustan GrabCut-algoritmilla.

    Olettaa, että kohde on lähellä kuvan keskiötä (passikuva, rintakuva).

    Args:
        img: BGR-kuva.

    Returns:
        (result, fg_mask):
          result   – BGR-kuva, tausta korvattu valkoisella.
          fg_mask  – Binäärimaski (0=tausta, 255=kohde).
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 5 % marginaali joka reunalta
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model,
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # GC_FGD = varma etuala, GC_PR_FGD = todennäköinen etuala
    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    result = img.copy()
    result[fg_mask == 0] = 255  # valkoinen tausta
    return result, fg_mask


# ===========================================================================
# METODI 1 – REUNANTUNNISTUS
# ===========================================================================

def method_edge_detection(img: np.ndarray,
                           low_thresh: int = 50,
                           high_thresh: int = 150,
                           min_line_length: int = 10) -> np.ndarray:
    """
    Tunnistaa kuvan reunat Canny-algoritmilla ja poistaa lyhyet häiriöviivat.

    Parametrit:
      - Pienennä kynnyksiä → enemmän reunoja (myös kohinaa).
      - Suurenna kynnyksiä → vain voimakkaimmat reunat.
      - min_line_length poistaa pienet pisteet ja lyhyet viivat.

    Args:
        img:             BGR- tai harmaasävykuva.
        low_thresh:      Canny:n alempi hystereesinkynnys (1–254).
        high_thresh:     Canny:n ylempi hystereesinkynnys (2–255).
        min_line_length: Pienin sallittu yhtenäinen pikselikomponentti.

    Returns:
        Binäärikuva: 0 = piirrä (reuna), 255 = tausta.
    """
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # Poista lyhyet komponentit
    if min_line_length > 1:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            edges, connectivity=8
        )
        clean = np.zeros_like(edges)
        for i in range(1, n_labels):  # 0 = tausta, ohitetaan
            if stats[i, cv2.CC_STAT_AREA] >= min_line_length:
                clean[labels == i] = 255
        edges = clean

    # Canny antaa 255=reuna; invertoi laserkonventioon (0=piirrä)
    return cv2.bitwise_not(edges)


# ===========================================================================
# METODI 2 – HATCH-VARJOSTUS
# ===========================================================================

def method_hatch_shading(img: np.ndarray,
                          levels: int = 8,
                          line_spacing: int = 6,
                          angle_deg: float = 0.0) -> np.ndarray:
    """
    Toistaa harmaasävyt tiheämpinä tai harvempina viivoina (hatch-varjostus).

    Toimintaperiaate:
      1. Posterisoi kuva → rajoitettu sävymäärä.
      2. Jokaisen pikselin tummuus (0–1) määrää paikallisen viivavälin:
         spacing = max_spacing × (1 − tummuus)
         Tumma → pieni väli → tiheät viivat.
         Vaalea → suuri väli → harvat viivat.
      3. Kulma toteutetaan projisoimalla piste (x·cosA + y·sinA) -akselille.

    KORJATTU BUG: Vanha "nopea" versio laski koko riville yhden
    keskiarvoviilavälin → sarakekohtainen vaihtelu katosi.
    Nyt jokainen pikseli lasketaan erikseen, jolloin kolumnit toimivat.

    Args:
        img:          BGR- tai harmaasävykuva.
        levels:       Posterisaatiotasojen määrä (2–32).
        line_spacing: Maksimiviivaväli pikseleinä (harvimmat alueet).
        angle_deg:    Viivojen kulma asteina (0 = vaaka, 90 = pysty).

    Returns:
        Binäärikuva: 0 = viiva, 255 = tausta.
    """
    gray = to_gray(img)
    post = posterize(gray, levels).astype(np.float32) / 255.0
    h, w = gray.shape
    result = np.ones((h, w), dtype=np.uint8) * 255

    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))

    for y in range(h):
        for x in range(w):
            darkness = 1.0 - post[y, x]          # 0=vaalea, 1=tumma
            spacing = max(1, int(line_spacing * (1.0 - darkness)))
            proj = int(x * cos_a + y * sin_a)     # projisointilinja
            if proj % spacing == 0:
                result[y, x] = 0

    return result


# ===========================================================================
# METODI 3 – RASTEROINTI (VAKIOKOKO, ETÄISYYS VAIHTELEE)
# ===========================================================================

def method_raster_spacing(img: np.ndarray,
                           dot_radius: int = 2,
                           grid_size: int = 10) -> np.ndarray:
    """
    Sanomalehtimäinen rasterointi Floyd-Steinberg-ditheringillä.

    Pisteet ovat aina samankokoisia; niiden esiintymistiheys kasvaa
    tummuuden myötä. Tumma alue = tiheä pisteistö, vaalea = harva.

    Toteutus:
      1. Pienennä kuva "dithering-resoluutioon" (1 px = 1 solukoko).
      2. Floyd-Steinberg-dithering → jokainen ruutu on 0 tai 255.
      3. Piirrä ympyrä jokaisen tumman ruudun kohdalle.

    KORJATTU BUG: Alkuperäisessä v1:ssä threshold-muuttuja laskettiin
    mutta jätettiin käyttämättä, ja modulo-logiikka oli rikki.
    Nyt käytetään oikeaa FS-dithering-algoritmia.

    Args:
        img:        BGR- tai harmaasävykuva.
        dot_radius: Pisteen säde pikseleinä.
        grid_size:  Ruudukon solun koko pikseleinä.

    Returns:
        Binäärikuva: 0 = piste, 255 = tausta.
    """
    gray = to_gray(img).astype(np.float32)
    h, w = gray.shape
    result = np.ones((h, w), dtype=np.uint8) * 255

    small_h = max(1, h // grid_size)
    small_w = max(1, w // grid_size)
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Floyd-Steinberg dithering:
    # Kvantisointivirhe levitetään neljään naapuriin kertoimilla
    # 7/16 (oikea), 3/16 (vasen-alas), 5/16 (alas), 1/16 (oikea-alas)
    err = small.copy()
    dithered = np.zeros((small_h, small_w), dtype=np.uint8)

    for y in range(small_h):
        for x in range(small_w):
            old_val = err[y, x]
            new_val = 255.0 if old_val > 127 else 0.0
            dithered[y, x] = int(new_val)
            quant_err = old_val - new_val

            if x + 1 < small_w:
                err[y, x + 1] += quant_err * (7.0 / 16)
            if y + 1 < small_h:
                if x > 0:
                    err[y + 1, x - 1] += quant_err * (3.0 / 16)
                err[y + 1, x] += quant_err * (5.0 / 16)
                if x + 1 < small_w:
                    err[y + 1, x + 1] += quant_err * (1.0 / 16)

    # Piirrä ympyräpisteet alkuperäisen kuvan koordinaateissa
    for sy in range(small_h):
        for sx in range(small_w):
            if dithered[sy, sx] == 0:
                cx = sx * grid_size + grid_size // 2
                cy = sy * grid_size + grid_size // 2
                cv2.circle(result, (cx, cy), dot_radius, 0, -1)

    return result


# ===========================================================================
# METODI 4 – RASTEROINTI (ETÄISYYS VAKIO, KOKO VAIHTELEE)
# ===========================================================================

def method_raster_size(img: np.ndarray,
                        grid_size: int = 10,
                        max_radius_ratio: float = 0.46) -> np.ndarray:
    """
    Rasterointi, jossa pisteiden koko skaalautuu tummuuden mukaan.

    Laser-käytössä: iso piste = enemmän energiaa = tummempi poltto.
    Piirturilla: iso täytetty ympyrä.

    Pisteet ovat tasavälisessä grid_size × grid_size px ruudukossa.
    Ruudun keskiarvo-tummuus → pistesäde:
      r = darkness × max_r

    Args:
        img:             BGR- tai harmaasävykuva.
        grid_size:       Solukoko pikseleinä.
        max_radius_ratio: Suurimman pisteen säde / solukoko.
                          0.46 = lähes koko solu täyttyy.

    Returns:
        Binäärikuva: 0 = piste, 255 = tausta.
    """
    gray = to_gray(img)
    h, w = gray.shape
    result = np.ones((h, w), dtype=np.uint8) * 255
    max_r = max(1, int(grid_size * max_radius_ratio))

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            block = gray[y:y + grid_size, x:x + grid_size]
            if block.size == 0:
                continue
            darkness = 1.0 - np.mean(block) / 255.0
            r = int(darkness * max_r)
            if r >= 1:
                cx = x + grid_size // 2
                cy = y + grid_size // 2
                cv2.circle(result, (cx, cy), r, 0, -1)

    return result


# ===========================================================================
# METODI 5 – SINIAALTORASTEROINTI
# ===========================================================================

def method_sine_wave(img: np.ndarray,
                      line_gap: int = 12,
                      amplitude_scale: float = 1.0,
                      direction: str = "horizontal") -> np.ndarray:
    """
    Piirtää siniaaltoviivoja, joiden amplitudi kasvaa tummuuden mukaan.

    Vaalea alue → aalto pysyy lähellä keskilinjaa (pieni amplitudi).
    Tumma alue  → aalto heilahtelee leveästi (suuri amplitudi).

    KORJATTU BUG: Vanha parametrinimi "frequency" viittasi rivivälin,
    mikä oli harhaanjohtavaa. Nyt: line_gap = rivien välinen etäisyys (px).
    Lisäksi vaihe lasketaan suhteessa kuvan koko leveyteen/korkeuteen
    → yksi täysi aaltojakso näkyvissä kerrallaan (selkeämpi tulos).

    Args:
        img:             BGR- tai harmaasävykuva.
        line_gap:        Aaltoviivojen välinen etäisyys pikseleinä.
        amplitude_scale: Amplitudin skaalaustekijä (1.0 = normaali).
        direction:       "horizontal" tai "vertical".

    Returns:
        Binäärikuva: 0 = viiva, 255 = tausta.
    """
    gray = to_gray(img).astype(np.float32) / 255.0  # 0=tumma, 1=vaalea
    h, w = gray.shape
    result = np.ones((h, w), dtype=np.uint8) * 255
    half_gap = line_gap // 2

    if direction == "horizontal":
        for y_center in range(half_gap, h, line_gap):
            for x in range(w):
                y_sample = min(y_center, h - 1)
                darkness = 1.0 - float(gray[y_sample, x])
                amp = half_gap * darkness * amplitude_scale

                # Yksi täysi aalto kuvan leveydellä
                phase = (x / max(w - 1, 1)) * 2 * math.pi
                offset = int(round(amp * math.sin(phase)))

                py = max(0, min(h - 1, y_center + offset))
                result[py, x] = 0
    else:
        for x_center in range(half_gap, w, line_gap):
            for y in range(h):
                x_sample = min(x_center, w - 1)
                darkness = 1.0 - float(gray[y, x_sample])
                amp = half_gap * darkness * amplitude_scale
                phase = (y / max(h - 1, 1)) * 2 * math.pi
                offset = int(round(amp * math.sin(phase)))
                px = max(0, min(w - 1, x_center + offset))
                result[y, px] = 0

    return result


# ===========================================================================
# METODI 6 – SYVYYSKARTTA (Z-KAIVERRUS)
# ===========================================================================

def method_depth_map(img: np.ndarray, levels: int = 16) -> np.ndarray:
    """
    Muuntaa harmaasävyn posterisoiduksi syvyyskartaksi CNC-kaivertimelle.

    Posterisaatio rajaa syvyystasot, jotta kaiverrin ei yritä
    toteuttaa jatkuvaa liukukaistaa.

    Tulkinta G-koodissa:
      arvo=0   (musta)   → syvin kohta (suurin Z-syvyys).
      arvo=255 (valkoinen) → ei kaivertaa.

    Käytetään GCodeGenerator.from_grayscale_image()-metodin kanssa.

    Args:
        img:    BGR- tai harmaasävykuva.
        levels: Kaivertussyvyystasojen määrä (2–64).
    """
    gray = to_gray(img)
    return posterize(gray, levels)


# ===========================================================================
# G-KOODIN GENEROINTI
# ===========================================================================

class GCodeGenerator:
    """
    Muuntaa prosessoidun kuvan G-koodiksi laser- tai piirturityöstöä varten.

    Tuetut toimilaitteet:
      "laser"   – Tehonsäätö M3 S<arvo> / M3 S0 (GRBL/Marlin -yhteensopiva).
      "plotter" – Kynä ylös/alas Z-akselilla.
      "cnc"     – Kaiverrus Z-syvyydellä (käytä from_grayscale_image).

    Serpentiinikulku (boustrophedon):
      Parisilta riviltä vasemmalta oikealle, parittomilta oikealta vasemmalle.
      Minimoi siirtoliikkeet ja lyhentää kokonaistyöaikaa.

    Koordinaatisto:
      X kasvaa oikealle, Y kasvaa ylöspäin (matematiikan konventio).
      Kuvan rivi 0 = korkein Y-koordinaatti G-koodissa.
    """

    def __init__(self,
                 device: str = "laser",
                 feed_rate: int = 1000,
                 laser_power_max: int = 1000,
                 z_up: float = 5.0,
                 z_down: float = 0.0,
                 pixel_size_mm: float = 0.1):
        """
        Args:
            device:          "laser", "plotter" tai "cnc".
            feed_rate:       Liikenopeus mm/min.
            laser_power_max: Laserin S-arvo täydellä teholla (255 tai 1000).
            z_up:            Z-korkeus siirtymissä (turvakorkeus, mm).
            z_down:          Z-korkeus piirtämisessä (kynä alas, mm).
            pixel_size_mm:   Yhden pikselin fyysinen koko millimetreinä.
        """
        self.device = device
        self.feed_rate = feed_rate
        self.laser_power_max = laser_power_max
        self.z_up = z_up
        self.z_down = z_down
        self.pixel_size = pixel_size_mm

    # ------------------------------------------------------------------
    # Yksityiset apumetodit
    # ------------------------------------------------------------------

    def _header(self) -> list:
        """
        G-kooditiedoston aloitusrivit.

        KORJATTU BUG: Vanha koodi asetti F-arvon G0-rivin yhteydessä
        mutta ei erikseen G1:lle. Nyt molemmat nopeudet asetetaan
        eksplisiittisesti ennen ensimmäistä liikekäskyä.
        """
        lines = [
            "; === LaserPlotter G-koodi ===",
            "G21         ; Yksikkö: millimetrit",
            "G90         ; Absoluuttikoordinaatit",
            f"G0 F{self.feed_rate}  ; Siirtymisnopeus mm/min",
            f"G1 F{self.feed_rate}  ; Piirtonopeus mm/min",
        ]
        if self.device == "laser":
            lines += [
                "M5          ; Varmista laser pois",
                "G28 X Y     ; Kotiasema",
                "M3 S0       ; Laser valmiustilaan (teho 0)",
            ]
        else:
            lines += [
                f"G0 Z{self.z_up:.3f}  ; Työkalu ylös",
                "G28 X Y     ; Kotiasema",
            ]
        return lines

    def _footer(self) -> list:
        """G-kooditiedoston lopetusrivit."""
        lines = []
        if self.device == "laser":
            lines += ["M5          ; Laser pois"]
        else:
            lines += [f"G0 Z{self.z_up:.3f}  ; Työkalu ylös"]
        lines += [
            "G0 X0 Y0    ; Paluu alkupisteeseen",
            "M2          ; Ohjelma loppu",
        ]
        return lines

    def _px_to_mm(self, x: int, y: int, img_height: int) -> tuple:
        """
        Muuntaa pikselikoordinaatit millimetreiksi.
        Y-akseli käännetään: kuvan ylärivi (y=0) → suurin Y G-koodissa.
        """
        xmm = x * self.pixel_size
        ymm = (img_height - 1 - y) * self.pixel_size
        return xmm, ymm

    # ------------------------------------------------------------------
    # Julkiset generointimetodit
    # ------------------------------------------------------------------

    def from_binary_image(self, img: np.ndarray) -> str:
        """
        Muuntaa binäärikuvan G-koodiksi (metodit 1, 2, 3, 4, 5).

        Piirtologiikka:
          pikseliarvo 0   → piirrä (laser päälle / kynä alas)
          pikseliarvo 255 → ohita (laser pois / kynä ylös)

        Yhtenäinen mustien pikselien jono tulostetaan yhtenä G1-viivana,
        ei erillisinä pistekomennot → huomattavasti lyhyempi G-koodi.

        Args:
            img: Binäärikuva (uint8), arvot 0 tai 255.

        Returns:
            G-koodimerkkijono.
        """
        h, w = img.shape[:2]
        lines = self._header()
        pen_down = False

        for y in range(h):
            # Serpentiinikulku: parillinen rivi → vasemmalta, pariton → oikealta
            x_range = range(w) if y % 2 == 0 else range(w - 1, -1, -1)

            for x in x_range:
                is_dark = (img[y, x] == 0)
                xmm, ymm = self._px_to_mm(x, y, h)

                if is_dark:
                    if not pen_down:
                        # Siirry piirtokohtaan ja aloita
                        lines.append(f"G0 X{xmm:.3f} Y{ymm:.3f}")
                        if self.device == "laser":
                            lines.append(f"M3 S{self.laser_power_max}")
                        else:
                            lines.append(f"G1 Z{self.z_down:.3f}")
                        pen_down = True
                    lines.append(f"G1 X{xmm:.3f} Y{ymm:.3f}")

                else:
                    if pen_down:
                        # Lopeta piirto
                        if self.device == "laser":
                            lines.append("M3 S0")
                        else:
                            lines.append(f"G0 Z{self.z_up:.3f}")
                        pen_down = False

        # Sulje viimeinen avoin viiva
        if pen_down:
            if self.device == "laser":
                lines.append("M3 S0")
            else:
                lines.append(f"G0 Z{self.z_up:.3f}")

        lines += self._footer()
        return "\n".join(lines)

    def from_grayscale_image(self, img: np.ndarray,
                              z_max_depth: float = 3.0) -> str:
        """
        Muuntaa harmaasävykuvan Z-kaivertus-G-koodiksi (metodi 6).

        Harmaasävy → kaivertussyvyys:
          arvo=0   (musta)   → syvyys = z_max_depth (syvin)
          arvo=255 (valkoinen) → syvyys = 0 (ei kaiverreta)

        Args:
            img:         Harmaasävykuva (uint8), tyypillisesti method_depth_map:n tulos.
            z_max_depth: Suurin kaivertussyvyys millimetreinä.
        """
        h, w = img.shape[:2]
        lines = self._header()

        for y in range(h):
            x_range = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
            for i, x in enumerate(x_range):
                depth = (1.0 - int(img[y, x]) / 255.0) * z_max_depth
                xmm, ymm = self._px_to_mm(x, y, h)
                zmm = -depth  # negatiivinen = syvemmälle materiaaliin

                if i == 0:
                    # Rivin alussa nosta työkalu ensin turvakorkeuteen
                    lines.append(f"G0 X{xmm:.3f} Y{ymm:.3f} Z{self.z_up:.3f}")
                lines.append(f"G1 X{xmm:.3f} Y{ymm:.3f} Z{zmm:.3f}")

        lines += self._footer()
        return "\n".join(lines)

    def from_dot_image(self, img: np.ndarray,
                        dot_power_map: np.ndarray = None,
                        dwell_ms: int = 50) -> str:
        """
        Muuntaa pistepohjaiseen kuvaan (metodi 4) sopivan G-koodin.

        Jokainen piste käsitellään erillisenä polttopisteeenä.

        KORJATTU BUG: Vanha koodi käytti "G1 X Y" samaan pisteeseen,
        mikä ei tuota palopistettä GRBL:ssä (nollapituinen liike = ohitetaan).
        Nyt käytetään G4 P<ms> (dwell = viipymä), joka on oikea tapa
        polttopisteen tekemiseen laserilla.

        Args:
            img:           Binäärikuva, pisteet arvolla 0.
            dot_power_map: Valinnainen tehokartta (uint8).
                           Säätää laserin tehon pistekohtaisesti.
            dwell_ms:      Viipymäaika millisekunteina per piste.
        """
        h, w = img.shape[:2]
        lines = self._header()

        # Etsi pisteet kontuureina → massakeskipiste = pisteen sijainti
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue  # Degeneroitunut kontouri

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            xmm, ymm = self._px_to_mm(cx, cy, h)

            # Teho: tehokartan arvo tai vakioteho
            if dot_power_map is not None:
                cy_s = min(cy, dot_power_map.shape[0] - 1)
                cx_s = min(cx, dot_power_map.shape[1] - 1)
                power = max(1, int(dot_power_map[cy_s, cx_s] / 255.0
                                   * self.laser_power_max))
            else:
                power = self.laser_power_max

            lines.append(f"G0 X{xmm:.3f} Y{ymm:.3f}")
            if self.device == "laser":
                lines.append(f"M3 S{power}")
                lines.append(f"G4 P{dwell_ms}  ; Polttoviipymä {dwell_ms} ms")
                lines.append("M3 S0")
            else:
                lines.append(f"G1 Z{self.z_down:.3f}")
                lines.append(f"G4 P{dwell_ms}  ; Viipymä {dwell_ms} ms")
                lines.append(f"G0 Z{self.z_up:.3f}")

        lines += self._footer()
        return "\n".join(lines)
