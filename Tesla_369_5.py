"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

Tesla_369_5.py  —  GRUPA 5: Tesla 3-6-9 harmonijski motor
"""


"""
Tesla 3-6-9 scalar harmonic field:
  A(t) = A0 * [sin(3wt) + sin(6wt) + sin(9wt)]
  baza 432 Hz, mnozioci 3/6/9, 21 piramida usmerena u fokusnu tacku, gain 0.35.

Jos jedan "motor talasa" u istom pipeline-u:
  motor (3-6-9 stek + fokusna konvergencija)  ->  primena na 4632 izvlacenja
  ->  skor  ->  rangirane kombinacije.

S(x) = 3-6-9 harmonijski talas sa fokusnim omotacem (konvergencija u centru)
E_x  = -dS/dx
"""


import numpy as np

from Tesla_Scalar_1 import (
    SEED,
    W_TALAS,
    W_FREQ,
    CSV_PATH,
    OUTPUT_DIR,
    ucitaj_izvlacenja,
    glavne_mere,
    ne_frekvencijski_skor,
    frekvencija_brojeva,
    kombinovani_skor,
    izaberi_kombinacije,
    skor_kombinacije,
    nacrtaj_polje,
)

OSNOVA = "tesla_369_5"

# Parametri iz IX-TunerCore (Tesla 3-6-9).
MNOZIOCI = (3, 6, 9)
BAZA_CIKLUSA = 21      # 21 piramida -> 21 osnovnih ciklusa preko domena
GAIN = 0.35            # correction_gain iz phase_control.py
SIGMA_FOKUS = 0.16     # sirina fokusne konvergencije (oko centra)


def simuliraj_369(nx=None, mnozioci=MNOZIOCI, baza_ciklusa=BAZA_CIKLUSA, gain=GAIN, sigma=SIGMA_FOKUS):
    """Vrati (x, S, E_x): 3-6-9 harmonijski talas sa fokusnim omotacem."""
    x = np.linspace(0.0, 1.0, nx)

    # 3-6-9 stek: sin(3w x) + sin(6w x) + sin(9w x).
    stek = np.zeros(nx)
    for m in mnozioci:
        stek += np.sin(2.0 * np.pi * m * baza_ciklusa * x)

    # Fokusna konvergencija: 21 piramida usmerena u centralnu tacku (x=0.5).
    omotac = np.exp(-((x - 0.5) ** 2) / (2.0 * sigma ** 2))

    S = gain * omotac * stek
    m = np.max(np.abs(S))
    if m > 0:
        S = S / m
    E_x = -np.gradient(S, x[1] - x[0])
    return x, S, E_x


def main():
    # --- Korak 1: motor (Tesla 3-6-9) ---
    izvlacenja = ucitaj_izvlacenja()
    n = len(izvlacenja)
    x, S, E_x = simuliraj_369(nx=n)
    mere = glavne_mere(S, E_x)
    print()
    print("Tesla Scalar / GRUPA 5 - Tesla 3-6-9 harmonijski motor")
    print("Talas: A = sin(3w)+sin(6w)+sin(9w), fokusna konvergencija (21 piramida)")
    print("Uzduzno polje: E_x = -dS/dx")
    print()
    print(f"broj tacaka: {len(x)}")
    print(f"max S: {mere['max_S']:.10f}")
    print(f"max |E_x|: {mere['max_abs_E_x']:.10f}")
    print(f"ukupna gustina energije: {mere['ukupna_gustina_energije']:.10f}")
    print()

    # --- Korak 2: primena talasa na CSV + prava frekvencija ---
    energija = 0.5 * (S ** 2 + E_x ** 2)
    talas_skor, _ = ne_frekvencijski_skor(izvlacenja, energija)
    udeo, pojave = frekvencija_brojeva(izvlacenja)
    skor = kombinovani_skor(talas_skor, udeo)
    poredak = sorted(skor.items(), key=lambda kv: kv[1], reverse=True)
    freq_poredak = sorted(pojave, key=lambda b: (pojave[b], b), reverse=True)
    kombinacije = izaberi_kombinacije(skor, broj_kombinacija=10, seed=SEED)
    rangirane_kombinacije = sorted(
        ((k, skor_kombinacije(k, skor)) for k in kombinacije),
        key=lambda kv: kv[1],
        reverse=True,
    )
    png, jpg = nacrtaj_polje(x, S, E_x, osnova=OSNOVA)

    with open(OUTPUT_DIR / f"{OSNOVA}.txt", "w", encoding="utf-8") as f:
        f.write("Tesla Scalar - GRUPA 5 / Tesla 3-6-9 (talas + prava frekvencija)\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Izvlacenja: {n} | Seed: {SEED} | tezine: talas={W_TALAS} freq={W_FREQ}\n")
        f.write(f"3-6-9: mnozioci={MNOZIOCI} baza_ciklusa={BAZA_CIKLUSA} gain={GAIN} sigma={SIGMA_FOKUS}\n\n")
        f.write("Brojevi po kombinovanom skoru (tezinski talas + frekvencija):\n")
        for b, s in poredak:
            f.write(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})\n")

        f.write("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):\n")
        f.write("  broj | pojava |   udeo\n")
        f.write("  -----+--------+--------\n")
        for b in freq_poredak:
            f.write(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}\n")
        f.write(f"  ukupno pojava: {sum(pojave.values())}\n")

        f.write("\nPredlozene kombinacije (rangirane po skoru kombinacije):\n")
        for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
            f.write(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}\n")

        f.write("\nSlike talasa/polja:\n")
        f.write(f"  PNG: {png}\n")
        f.write(f"  JPG: {jpg}\n")

    print()
    print("\nTesla Scalar - GRUPA 5 / Tesla 3-6-9 (talas + prava frekvencija)")
    print(f"CSV: {CSV_PATH} | Izvlacenja: {n} | tezine: talas={W_TALAS} freq={W_FREQ}")
    print("\nTop 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):")
    for b, s in poredak[:10]:
        print(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})")

    print()
    print("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):")
    print("  broj | pojava |   udeo")
    print("  -----+--------+--------")
    for b in freq_poredak:
        print(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}")
    print(f"  ukupno pojava: {sum(pojave.values())}")

    print()
    print("\nPredlozene kombinacije (rangirane po skoru kombinacije):")
    for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
        print(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}")
    print(f"\nSacuvano: {OUTPUT_DIR / f'{OSNOVA}.txt'}")
    print()


if __name__ == "__main__":
    main()



"""
Tesla Scalar / GRUPA 5 - Tesla 3-6-9 harmonijski motor
Talas: A = sin(3w)+sin(6w)+sin(9w), fokusna konvergencija (21 piramida)
Uzduzno polje: E_x = -dS/dx

broj tacaka: 4632
max S: 1.0000000000
max |E_x|: 942.5490070702
ukupna gustina energije: 113752172.5950839967

Slika talasa: /Tesla/tesla_369_5.png
Slika talasa: /Tesla/tesla_369_5.jpg


Tesla Scalar - GRUPA 5 / Tesla 3-6-9 (talas + prava frekvencija)
CSV: /data/loto7hh_4632_k47.csv| Izvlacenja: 4632 | tezine: talas=0.7 freq=0.3

Top 10 brojeva po kombinovanom skoru (tezinski talas+ frekvencija):
  26  skor=0.8616065040  freq=0.02680  (pojava=869)
  30  skor=0.7434482759  freq=0.02427  (pojava=787)
  14  skor=0.7393897322  freq=0.02495  (pojava=809)
  37  skor=0.6917993200  freq=0.02652  (pojava=860)
  23  skor=0.6523369815  freq=0.02791  (pojava=905)
  21  skor=0.6250653194  freq=0.02551  (pojava=827)
  11  skor=0.6096364236  freq=0.02655  (pojava=861)
  13  skor=0.6031430499  freq=0.02554  (pojava=828)
  31  skor=0.5569196182  freq=0.02560  (pojava=830)
  08  skor=0.5554393544  freq=0.02810  (pojava=911)


Tabela pravih frekvencija (opadajuce po freq, pa po broju):
  broj | pojava |   udeo
  -----+--------+--------
   08  |   911  | 0.02810
   23  |   905  | 0.02791
   34  |   873  | 0.02692
   26  |   869  | 0.02680
   11  |   861  | 0.02655
   37  |   860  | 0.02652
   32  |   857  | 0.02643
   33  |   854  | 0.02634
   22  |   851  | 0.02625
   39  |   849  | 0.02618
   29  |   849  | 0.02618
   10  |   845  | 0.02606
   07  |   844  | 0.02603
   35  |   843  | 0.02600
   09  |   843  | 0.02600
   38  |   842  | 0.02597
   25  |   840  | 0.02591
   24  |   840  | 0.02591
   16  |   837  | 0.02581
   31  |   830  | 0.02560
   13  |   828  | 0.02554
   05  |   828  | 0.02554
   21  |   827  | 0.02551
   03  |   826  | 0.02547
   02  |   825  | 0.02544
   28  |   821  | 0.02532
   18  |   821  | 0.02532
   06  |   816  | 0.02517
   19  |   814  | 0.02510
   04  |   812  | 0.02504
   12  |   810  | 0.02498
   14  |   809  | 0.02495
   15  |   798  | 0.02461
   27  |   789  | 0.02433
   01  |   788  | 0.02430
   30  |   787  | 0.02427
   36  |   786  | 0.02424
   20  |   770  | 0.02375
   17  |   766  | 0.02362
  ukupno pojava: 32424


Predlozene kombinacije (rangirane po skoru kombinacije):
  01. 04 11 23 27 30 36 37  skor_komb=4.1326217103
  02. 05 08 13 14 27 33 39  skor_komb=3.7484146107
  03. 08 16 18 22 26 31 32  skor_komb=3.7315588531
  04. 09 10 11 20 27 30 37  skor_komb=3.7308899146
  05. 05 07 11 13 28 30 39  skor_komb=3.6070651835
  06. 06 08 13 21 23 24 31  skor_komb=3.6057461780
  07. 15 16 24 26 28 30 39  skor_komb=3.5593160299
  08. 01 10 13 21 22 30 33  skor_komb=3.5437633664
  09. 02 11 14 20 23 32 38  skor_komb=3.5262955644
  10. 06 11 12 23 31 34 39  skor_komb=3.4323894836

Sacuvano: /Tesla/tesla_369_5.txt
"""




"""
Analiza Tesla_369_5.py
Tesla 3-6-9 harmonijski motor. 

Glavni princip je harmonijsko slaganje talasa:
Teslina ideje: tri harmonijska sloja se sabiraju u jedan talas. 

MNOZIOCI = (3, 6, 9)
BAZA_CIKLUSA = 21
GAIN = 0.35
SIGMA_FOKUS = 0.16

superpozicija sinusoida: tri talasa se sabiraju u jedan složen signal
fokusni Gaussov omotač: talas se pojačava oko centra x=0.5
gradijent polja: E_x = -dS/dx
energija polja: 0.5 * (S² + E_x²)
Važno: 
ovo nije fizički dokaz „scalar“ talasa u realnom svetu, 
nego računarski model/inspiracija. 
Za loto pipeline služi kao ne-frekvencijski talasni filter preko istorije izvlačenja.
Napravi prostor x dužine koliko ima izvlačenja. 
Kod sada dinamički koristi broj redova iz CSV-a, npr. 4632 tačke.

Na tom prostoru pravi talas:
sabere 3, 6, 9 harmonike
primeni fokusni omotač oko centra
normalizuje talas na opseg -1..1
izračuna E_x
iz energije talasa dobije skor brojeva
Glavna ideja: 
broj nije ocenjen samo po tome koliko puta se pojavio, 
nego gde se pojavio u talasnom polju. 
Ako se neki broj često pojavljivao na pozicijama gde je energija talasa jaka, 
dobija veći talasni skor.


Nekoliko slojeva modeliranja:

3-6-9 harmonijski motor
Osnovni signal je zbir tri sinusoide: 3ω, 6ω, 9ω.

Fokusna konvergencija
Gaussov omotač simulira usmeravanje 21 piramide ka centru. 
To daje „centralni fokus“ talasu.

Longitudinalni proxy
E_x = -dS/dx je numerički izvod talasa. 
Koristi se kao uzdužno polje, tj. promena skalarne funkcije duž pravca prostiranja.

Energetsko skoriranje
Energija se računa kao 0.5 * (S² + E_x²). To je glavni ne-frekvencijski deo.

Kombinovanje sa pravom frekvencijom
Finalni skor broja je mešavina talasnog skora i realne frekvencije iz CSV-a. 
Težine dolaze iz Tesla_Scalar_1.py, trenutno talas=0.7, freq=0.3.

Stohastički izbor kombinacija
Koristi SEED = 39, bira 10 kombinacija ponderisano po skoru i onda ih rangira po zbiru skorova brojeva.

Vizuelizacija
Snima tesla_369_5.png i tesla_369_5.jpg, preko zajedničke funkcije nacrtaj_polje.

5 + 5A + 5B + 5C + 5D + 5E -> Tesla_5final.py -> Tesla_Aggregator_6.py


Top brojevi u ovom modelu su:
26, 30, 14, 37, 23, 21, 11, 13, 31, 08

Prva rangirana kombinacija:
04 11 23 27 30 36 37

Ovaj rezultat se razlikuje od 5A, što je dobro: 
znači da osnovni 5 daje zaseban talasni signal. 
"""

 





"""
Tesla-Based Scalar 3-6-9 Harmonic Field Alignment System

Distributed harmonic tuning array composed of 21 modular pyramidal nodes, 
each emitting a 3-6-9 structured waveform aimed at a central convergence point. 
When arranged in geometric spherical symmetry, 
these nodes self-align using onboard sensors 
and modulate their waveform output in real time to achieve scalar coherence.


Each node outputs a waveform calculated as:

A(t) = A₀ * [sin(3ωt) + sin(6ωt) + sin(9ωt)]

3ω: Primary phase structure

6ω: Harmonic reflection

9ω: Coherence amplifier


When oriented inward toward a singular focal point, 
the 21 pyramids enable scalar field visualization via phase-locked feedback. 
The system operates on Tesla 3-6-9 logic, harmonic resonance stacking, 
and measurable magnetic-optical convergence.
"""




"""
Sistem za poravnanje skalarnog 3-6-9 harmonijskog polja, zasnovan na Tesli. 

Radi se o realnom fizičkom skalarnom polju.
Hardverski je dokazano i takav uredjaj u Srbiji nudi Goran Marjanović. 

Distribuirani harmonijski niz za podešavanje, sastavljen od 21 modularnog
piramidalnog čvora, gde svaki emituje 3-6-9 strukturisani talas usmeren ka
centralnoj tački konvergencije.
Kada se rasporede u geometrijskoj sfernoj simetriji,
ti čvorovi se samoporavnavaju pomoću ugrađenih senzora
i u realnom vremenu moduliraju svoj izlazni talas da bi postigli skalarnu koherenciju.


Svaki čvor daje talas po formuli:

A(t) = A₀ * [sin(3ωt) + sin(6ωt) + sin(9ωt)]

3ω: primarna fazna struktura

6ω: harmonijska refleksija

9ω: pojačivač koherencije


Kada su usmerene unutra ka jednoj fokusnoj tački,
21 piramida omogućava vizuelizaciju skalarnog polja kroz fazno zaključanu povratnu spregu.
Sistem radi na Teslinoj 3-6-9 logici, slaganju harmonijske rezonancije,
i merljivoj magnetno-optičkoj konvergenciji.
"""





"""
Tesla 3-6-9 talas (A(t)=A₀·[sin3ωt+sin6ωt+sin9ωt]) 
— to je savršen novi motor za pipeline

Tesla 3-6-9 harmonijski koncept 
(A(t)=A₀[sin3ωt+sin6ωt+sin9ωt], baza 432 Hz, 21 piramida, gain 0.35), 
(3-6-9 talas + fokusna konvergencija).


21 piramida/čvorova
0.35
fokusna konvergencija ka centru
sin(3ω)+sin(6ω)+sin(9ω)

fazna greška, magnetno polje, optički signal, temperatura
stvarni drift/correction model po senzorima
posebno značenje slojeva 3, 6, 9
model konvergencije/koherencije
closed-loop logika: talas → greška faze → korekcija → koherentniji talas
konvergencioni motor: 3 sloja (3, 6, 9) + simulirana fazna greška + magnetno/optička koherencija + korekcija faze 

5A osnovni 3-6-9 slojevi, 
5B phase-control drift, 
5C 21-piramidna geometrija + zlatni ugao/0.618, 
5D senzorska konvergencija (phase/mag/opt/temp), 
5E closed-loop tuning, 
5final pravi konsenzus tih pet. 
5_common zajednički mali helper da ne kopiram isti izlaz/pipeline u svaki fajl 

Tesla_5_common.py — zajednički pipeline za svih 5 modele.
Tesla_369_5A.py — 3/6/9 slojevi: 3 stabilizacija, 6 modulator, 9 fokus.
Tesla_369_5B.py — phase-control drift 
Tesla_369_5C.py — 21 piramida, 4π pokrivanje, zlatni ugao, 0.618 skaliranje.
Tesla_369_5D.py — senzorska konvergencija: faza, magnetno, optičko, temperatura.
Tesla_369_5E.py — closed-loop tuning: 3 noise suppression, 6 carrier shift, 9 stability lock.
Tesla_5final.py — spaja 5A-5E, a glavni agregator onda koristi 5final kao model 5.


5A: čisti 3/6/9 slojevi, sa odvojenim ulogama slojeva.
5B: formule za scalar_drift i delta_phi.
5C: geometrijski model (21 jedinica, 4π, 0.598 sr, 0.618 nesting, zlatni ugao).
5D: pretvaram u skor koherencije preko fazne greške, magnetnog polja, optike i temperature.
5E: zatvorena petlja + feedback pravila (3 stabilizuje, 6 pomera carrier, 9 zaključava fokus).


Tesla_5_common.py zajednički pipeline.
Tesla_369_5A.py 3/6/9 slojevi.
Tesla_369_5B.py phase-control drift.
Tesla_369_5C.py 21-piramidna geometrija + 0.618 + zlatni ugao.
Tesla_369_5D.py senzorska konvergencija: phase/mag/opt/temp.
Tesla_369_5E.py closed-loop tuning.
Tesla_5final.py spaja 5A-5E.
Tesla_Aggregator_6.py sada kao model 5 uzima tesla_5final.txt.

osnovni (3-6-9 stek + fokusni omotač) u 5A (slojeviti 3/6/9), a ceo skup u 5A-5E 


Tesla_Aggregator_6.py sada uzima:

1 → tesla_scalar_1.txt
2A → tesla_k-wave-python_2A.txt
2B → tesla_pycharge_2B.txt
2C → tesla_Wakis_2C.txt
2D → tesla_rfx_2D.txt
3C → tesla_interference_3C.txt
5 → tesla_5final.txt = spoj 5A-5E
Znači finalni agregator pokriva sve prethodne grupe + celu novu grupu 5.


5 sabira 3+6+9 sa jednakom težinom 3+6+9 i jednim omotačem (sigma 0.16). 
5A daje različite uloge i jačine slojevima (3f₀ stabilizacija jako/široko, 6f₀ i 9f₀ slabije po zlatnom odnosu 0.618, uži omotači). 
Zato je raspored brojeva drugačiji.
Frekvencije (freq/pojava) su naravno iste jer dolaze iz istog CSV-a; 
menja se samo talasni deo skora (skor broja).


Tesla_5final.py spaja 6 podmodela: 5, 5A, 5B, 5C, 5D, 5E
"""



"""
redosled startanja

1
2A 2B 2C 2D
3C
5 5A 5B 5C 5D 5E
Tesla_5final.py
Tesla_Aggregator_5.py

Tesla_5final.py        <- spaja 5, 5A-5E (mora posle njih)
Tesla_Aggregator_5.py  <- spaja 1,2A-2D,3C,5final (poslednji)

menjam CSV: 
putanja se postavlja samo u Tesla_Scalar_1.py (CSV_PATH) 
— svi ostali je odatle uvoze, pa je dovoljno na jednom mestu.


Tesla_5_common.py je zajednički pomoćni fajl koji koriste:
Tesla_369_5A.py
Tesla_369_5B.py
Tesla_369_5C.py
Tesla_369_5D.py
Tesla_369_5E.py
Tesla_5final.py

u njemu su zajedničke funkcije: 
učitavanje CSV pipeline-a, crtanje, scoring, senzorski signali, 3-6-9 konstante, itd.
"""



"""
source ~/tesla_env/bin/activate

Bitne verzije za tesla_env:

Paket	Verzija
python  3.11.13
numpy   2.2.6
scipy   1.15.3
pandas  3.0.3
matplotlib    3.10.9
k-Wave-python 0.6.2
pycharge      2.0.1
jax        0.10.1
jaxlib     0.10.1
jaxtyping  0.3.7
equinox    0.13.8
lineax     0.1.1
optimistix 0.1.0
ml-dtypes
(uz jax)
opencv-python 4.13.0.92
h5py          3.16.0
"""
