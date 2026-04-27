# Laser-piirturi / CNC-kaiverrin – Kuvamuokkain

## Vaatimukset

```
PyQt5>=5.15
opencv-python>=4.8
numpy>=1.24
Pillow>=10.0
scipy>=1.11
```

Asennus:
```bash
pip install PyQt5 opencv-python numpy Pillow scipy
```

## Käynnistys

```bash
cd laser_plotter
python main_app.py
```

## Moduulit

- **image_processor.py** – Kaikki kuvanmuokkausmetodit + G-koodigeneraattori
- **main_app.py** – PyQt5-käyttöliittymä

## Tuetut metodit

| # | Metodi | Kuvaus |
|---|--------|--------|
| 1 | Reunantunnistus | Canny-reuna + pienien viivojen poisto |
| 2 | Varjostettu viivapiirros | Hatch-varjostus + posterisaatio |
| 3 | Rasterointi (etäisyys) | Floyd-Steinberg dithering, vakiokoko |
| 4 | Rasterointi (koko/teho) | Pistekoko ~ tummuus, laserteho mukaan |
| 5 | Siniaaltorasterointi | Amplitudi ~ tummuus, vaakasuunta |
| 6 | Syvyyskartta | Harmaasävy → Z-kaiverrus |

## Tuetut toimilaitteet

- **Laser** – `M3 Sxxx` tehonsäätö, `M5` pois
- **X-Y-piirturi** – Z-akseli kynä ylös/alas
- **CNC-kaiverrin** – Z-syvyys harmaasävystä

## G-koodi-ominaisuudet

- Serpentiinikulku (boustrophedon) nopeuteen
- Absoluuttikoordinaatit (G90)
- Millimetrit (G21)
- Toimilaitekohtaiset käskyt
- Pisteiden G-koodi kontuuripohjaisesti

## Tallennukset

Kaikki kuvat (kamera + tulokset) tallennetaan automaattisesti
hakemistoon `~/laser_output/`.
