# Laser-piirturi / CNC-kaiverrin – Kuvamuokkain ja Simulaattori

Kaksi Python-ohjelmaa laser-piirturin ja CNC-kaivertimen käyttöön:

- **main_app.py** – Muuntaa valokuvan G-koodiksi laser-piirturille tai CNC-kaivertimelle
- **laser_simulator.py** – Visualisoi G-koodin animoituna simulaationa ennen ajoa laitteella

---

## Vaatimukset

```
Python 3.10 tai uudempi
PyQt5 >= 5.15
opencv-python >= 4.8
numpy >= 1.24
```

Asennus:

```bash
pip install PyQt5 opencv-python numpy
```

---

## main_app.py – Kuvamuokkain

Lataa valokuva, käsittele se yhdellä kuudesta metodista ja vie tulos G-koodina suoraan laitteelle.

### Käynnistys

```bash
python main_app.py
```

### Työnkulku

1. Paina **📂 Tiedostosta** tai **📷 Kameralta** ladataksesi kuvan
2. Valitse **Toimilaite** (laser / piirturi / CNC)
3. Valitse **Prosessointimetodi** (1–6)
4. Säädä metodin parametrit oikealle tasolle
5. Paina **▶ PROSESSOI KUVA** – tulos näkyy välittömästi oikealla
6. Paina **⚙ Generoi G-koodi** – koodi tallennetaan automaattisesti ja näkyy G-koodi-välilehdellä

### Prosessointimetodit

| # | Nimi | Kuvaus | Sopii |
|---|------|--------|-------|
| 1 | Reunantunnistus | Canny-algoritmi löytää reunat, lyhyet häiriöviivat suodatetaan pois | Piirturi, laser |
| 2 | Hatch-varjostus | Harmaasävyt toistetaan viivojen tiheydellä, posterisaatio ensin | Piirturi, laser |
| 3 | Rasterointi (etäisyys) | Floyd-Steinberg-dithering, pisteet vakiokokoisia | Laser, piirturi |
| 4 | Rasterointi (koko) | Pistekoko kasvaa tummuuden mukaan, laserilla teho skaalautuu | Laser |
| 5 | Siniaaltorasterointi | Siniaallon amplitudi vaihtelee tummuuden mukaan | Laser, piirturi |
| 6 | Syvyyskartta | Harmaasävy muunnetaan Z-kaiverrusarvoksi | CNC-kaiverrin |

### Toimilaitteiden G-koodi

**Laser (polttaminen)**
- `M3 S<arvo>` laser päälle (S = teho, GRBL: 0–1000, Marlin: 0–255)
- `M3 S0` laser pois
- Ei Z-akselia – vain X/Y-liike

**X-Y-piirturi (kynä)**
- `G1 Z0.000` kynä alas
- `G0 Z5.000` kynä ylös

**CNC-kaiverrin (metodi 6)**
- `G1 Z-3.000` jyrsi syvyyteen harmaasävyn mukaan

Kaikki G-koodi käyttää:
- `G21` – millimetrit
- `G90` – absoluuttikoordinaatit
- Serpentiinikulku (boustrophedon) siirtymien minimoimiseksi

### Parametrit

| Parametri | Sijainti | Kuvaus |
|-----------|----------|--------|
| Pikselikoko (mm/px) | Yhteiset asetukset | Yhden pikselin fyysinen koko – vaikuttaa G-koodin X/Y-arvoihin |
| Dwell-aika ms | Yhteiset asetukset | Viipymä per piste metodissa 4 (G4-käsky) |
| Syöttönopeus (mm/min) | Toimilaite | G0/G1 F-arvo |
| Laserin max-teho (S) | Toimilaite | M3 S-arvo täydellä teholla |
| Z turvakorkeus (mm) | Toimilaite | Z-korkeus siirtymissä |
| Piirtoalueen leveys/korkeus | Alue & Laite -välilehti | Skaalaa kuvan fyysiselle alueelle |

### Henkilökuvamoodi

Ruksaa **Henkilökuva – poista tausta** ennen prosessointia. GrabCut-algoritmi erottaa henkilön taustasta automaattisesti. Toimii parhaiten kun kohde on kuvan keskialueella.

### Tallennukset

Kaikki tiedostot tallentuvat automaattisesti `Output`-kansioon joka sijaitsee samassa hakemistossa kuin `main_app.py`:

| Tiedosto | Milloin syntyy |
|----------|----------------|
| `camera_YYYYMMDD_HHMMSS.jpg` | Kamerakuva |
| `result_<metodi>_YYYYMMDD_HHMMSS.png` | Prosessoinnin tulos |
| `gcode_<metodi>_<laite>_YYYYMMDD_HHMMSS.gcode` | G-koodi |

Tiedostonvalintaikkunat aukeavat ohjelmahakemistoon.

---

## laser_simulator.py – G-koodi-simulaattori

Visualisoi G-kooditiedoston animoituna lasersimulointina. Polttojälki piirtyy reaaliajassa kerroksittain: ulompi hehku, keltainen sisäkerros, tumma palanut ura ja valkoinen ydinviiva.

### Käynnistys

```bash
# Avaa tyhjänä
python laser_simulator.py

# Avaa suoraan G-kooditiedostolla
python laser_simulator.py polku/tiedosto.gcode
```

### Käyttö

1. Avaa G-kooditiedosto **📂 Avaa G-koodi** -painikkeella tai toolbarin kautta
2. Paina **▶ Käynnistä** tai `Space` aloittaaksesi animaation
3. Säädä **Nopeus**-liukusäätimellä kuinka monta segmenttiä piirretään per frame
4. Paina **⏸ Tauko** / `Space` pysäyttääksesi
5. Käytä **← →** näppäimiä askel kerrallaan eteenpäin tai taaksepäin
6. Paina **F** tai **⊡ Sovita** sovittaaksesi näkymän

### Näppäinoikopolut

| Näppäin | Toiminto |
|---------|----------|
| `Space` | Käynnistä / Tauko |
| `R` | Nollaa alkuun |
| `→` | Yksi askel eteenpäin |
| `←` | Yksi askel taaksepäin |
| `End` | Hyppää loppuun |
| `F` | Sovita näkymä |
| `+` / `-` | Zoom sisään / ulos |
| `Ctrl+O` | Avaa tiedosto |
| `Ctrl+S` | Vie PNG |

### Zoom ja panorointi

- **Rullahiiri** – zoomaa kursorin kohtaan
- **Keski-/vasenpainike + Alt + drag** – panoroi

### Näyttöasetukset

| Asetus | Kuvaus |
|--------|--------|
| Näytä siirtoreitit | Näyttää G0-siirtoliikkeet sinisenä katkoviivana |
| Näytä ruudukko | Millimetriruudukko taustalla |
| Näytä laserpää | Liikkuva ristikko ja hehkuefekti |
| Polton voimakkuus | Skaalaa kaikkien polttoviivojen tummuutta |
| Hehkun koko (px) | Laserpään hehkusäde pikseleinä |

### Tuetut G-koodi-käskyt

| Käsky | Kuvaus |
|-------|--------|
| `G0 X Y` | Siirtoliike (ei polteta) |
| `G1 X Y` | Piirtoliike (poltto päällä jos M3 aktiivinen) |
| `G4 P<ms>` | Dwell-viipymä (pisteen poltto) |
| `G21` | Millimetrit |
| `G28` | Kotiasema (siirto origoon) |
| `G90` / `G91` | Absoluutti / suhteellinen koordinaatisto |
| `M3 S<arvo>` | Laser päälle, teho S |
| `M5` | Laser pois |
| `M2` / `M30` | Ohjelma loppu |

G20 (tuumamoodi) ei ole tuettu – ohjelma varoittaa konsoliin ja jatkaa millimetreissä.

### Polttokuvan vienti

Paina **💾 Vie PNG** tai `Ctrl+S`. Tallennusikkuna avautuu `Output`-kansioon.

---

## Tiedostorakenne

```
laser_portrait/
├── main_app.py          # Kuvamuokkain
├── laser_simulator.py   # G-koodi-simulaattori
├── image_processor.py   # Kuvanmuokkausmoottori (molemmat käyttävät)
├── README.md            # Tämä tiedosto
└── Output/              # Luodaan automaattisesti
    ├── camera_*.jpg
    ├── result_*.png
    ├── gcode_*.gcode
    └── simulaatio.png
```

`image_processor.py` sisältää kaikki kuvanmuokkausmetodit ja G-koodigeneraattorin. Se ei sisällä käyttöliittymäkoodia eikä Qt-riippuvuuksia.

---

## Tyypilliset ongelmatilanteet

**Ohjelma ei löydä image_processor.py:tä**
Varmista että `main_app.py` ja `image_processor.py` ovat samassa kansiossa.

**Laser-simulaattori näyttää tyhjän ruudun**
Paina **F** tai **⊡ Sovita** – piirtoalue saattaa olla näkymän ulkopuolella.

**G-koodia ei generoidu**
Prosessoi kuva ensin painamalla **▶ PROSESSOI KUVA** ennen G-koodin generointia.

**GrabCut ei poista taustaa oikein**
Kohteen tulisi olla kuvan keskialueella eikä koskettaa reunoja.

**GRBL ei hyväksy G-koodia**
Tarkista laserin max-teho-asetus: GRBL käyttää yleensä arvoa 1000, jotkut laitteet 255.
