# Twee mechanismen, grondig uitgelegd (NL) — voor de greenlight

Dit document legt de twee "moeilijke" dynamieken van de scriptie stap voor stap uit, zodat je ze morgen vloeiend kunt navertellen:

1. **Experiment 3:** waarom de curves (bescherming aan vs. uit) vroeg verschillen, in het midden samenvallen, en **laat weer uiteenlopen** — de Keynesiaanse vraagversterking.
2. **Waarom de skill-wage premie in élk scenario daalt** van ~1,8 naar ~1,3 — het initialisatie-effect.

---

## Deel 1 — Experiment 3: de late divergentie en de Keynesiaanse vraagterugkoppeling

### Wat je in de figuren ziet (het te verklaren feit)

Twee runs, identiek op alles behalve component C7 (transitievergoeding + ketenregeling + vaste contracten aan of uit), gepaarde seeds. De permutatietest zegt: de trajecten verschillen significant op **~34–35% van de horizon**, en die significante massa zit **aan het begin én aan het einde** — het midden is insignificant (Figuren 31–32). Effectgrootte klein-tot-middel: dz ≈ 0,7 op Gini en werkloosheid, 0,9 op de premie.

De vraag die de commissie zal stellen: *waarom gaat de kloof laat weer open, terwijl de ontslagvergoedingsrem juist wegzakt?*

### Het verhaal in drie fasen

**Fase 1 — Het zaadje (vroeg in de horizon): de ontslagvergoedingsrem werkt.**

De transitievergoeding zit als eenmalige upfront-kostenpost S in de NPV: `NPV = −(I + S) + Σ besparingen`. Alleen vaste werknemers genereren S, en S schaalt met diensttijd (1/3 maandsalaris per dienstjaar). Twee dingen vallen bij de warme start samen:

1. De geërfde vaste werknemers hebben **hoge diensttijd** (gemiddeld ~6,5 jaar, tot 10). De verschuldigde vergoeding per taak piekt daardoor vroeg op **~€40** (Figuur 33).
2. De AI-huurprijs is **nog aan het dalen**, dus veel adoptiebeslissingen zijn nog *marginaal* — de NPV zit net boven of net onder de drempel.

Een klein wigje kan een marginale beslissing omgooien. Rekenvoorbeeld (§7.2.4): taak met complexiteit 3 → I = €116; drempel zonder S: 1,15 × 116 = €133; met S ≈ €28 wordt dat €161 (+21%), oftewel ~€0,90 extra vereiste besparing per periode. Klein, maar genoeg om precies de marginale taken te kantelen. **Gevolg:** de beschermde economie automatiseert in de eerste fase iets minder en start met lagere werkloosheid en ongelijkheid. → Vandaar het significante begin.

**Fase 2 — Het zaadje stopt met groeien (midden): de rem zakt weg, maar verdwijnt niet in de boekhouding.**

Zodra de AI-prijs zijn vloer bereikt (vroeg, en identiek in beide runs — C5 is exogeen), is automatiseren zó goedkoop dat het wigje van €28 vrijwel geen beslissing meer omgooit. Drie diagnoses sluiten het intuïtieve-maar-foute verhaal uit (dat de rem later opnieuw zou aantrekken doordat de ketenregeling een verse vaste werknemersvoorraad opbouwt):

- De verschuldigde vergoeding per taak **piekt vroeg, daalt en vlakt af** (Figuur 33) — geen heropbouw.
- De gemiddelde diensttijd van vaste werknemers **stabiliseert rond 9,5 jaar** omdat exogene separaties (δ = 1,3%/maand) de opbouw aftoppen (Figuur 34) — geen onbegrensde groei.
- Het aandeel vaste contracten en de conversiestroom zijn **vlak** over de tweede helft.

De per-stap automatiseringskloof stabiliseert op een klein, ruwweg constant verschil (~2 procentpunt van de beroepsbevolking). In het midden is het verschil daardoor te klein voor de simultane band. → Vandaar het insignificante midden.

**Fase 3 — De versterking (laat): de Keynesiaanse lus maakt van het kleine verschil een niveauverschil.**

Nu komt M1, de macroterugkoppeling. De causale keten, schakel voor schakel:

```
meer automatisering (protection-off)
   → minder werkenden → lagere loonsom Y
   → lagere vraagintercept:  A(t) = A₀ + γ·Y(t−1) + γπ·Π(t−1)   (γ=0,05 > γπ=0,008)
   → lagere prijs:  p = A − B·Q
   → marge (p − AC) krimpt
   → expansiepoort sluit:  bedrijven breiden alleen uit als AC < p
   → géén proactieve heraanname van werklozen
   → nog lagere loonsom → nog lagere vraag → ...  (zichzelf versterkende spiraal)
```

De diagnostiek laat exact deze lus zien:

- **Figuur 35:** het aantal expansiepoort-blokkades per stap stijgt steil in de protection-off-run en blijft vlak mét bescherming — dít is de handtekening van de lus.
- **Figuur 36:** het vraagniveau divergeert laat naar beneden in de protection-off-run.
- **Figuur 37:** de divergentie concentreert zich bij de **laaggeschoolden**: ~73% werkgelegenheid zonder bescherming tegen ~79% mét. Waarom juist zij? De gesloten poort onderdrukt precies de heraanname van de goedkoopste, meest automatiseerbare werknemers — zij staan vooraan in de rij die niet meer wordt bediend.

**De bewijsvoering dat het níet de rem of de AI-prijs is:** de werkloosheidskloof loopt op tot **5–6 procentpunt**, terwijl de stabiele automatiseringskloof maar **~2 punt** kan verklaren. Het verschil is *gemiste herabsorptie* via het vraagkanaal, geen nieuwe automatisering. En de AI-prijs ligt in deze hele fase plat op zijn vloer (in beide runs identiek), dus die kan het verschil niet aandrijven.

### De éénzinssamenvatting (memoriseren)

> "De transitievergoeding zaait vroeg een klein automatiseringsverschil; dat verschil groeit daarna niet meer, maar de Keynesiaanse vraagterugkoppeling — lagere loonsom → lagere vraag → lagere prijs → gesloten expansiepoort → geen heraanname — versterkt dat kleine vroege verschil tot een blijvend niveauverschil. Bescherming vertraagt dus niet alleen de reis, het verschuift de bestemming een beetje."

### De eerlijke kanttekeningen (zeg ze zelf, vóór de commissie)

- De **grootte** van het late niveauverschil hangt af van hoe dicht bedrijven bij de winstgevendheidsgrens opereren — en dat is een gevolg van de patrooncalibratie (A0.13). Het is bovendien **heavy-tailed over seeds**: enkele runs waarin de vraaglus doorschiet dragen onevenredig bij aan de late gemiddelde divergentie.
- Het late effect is een **bovengrens**: het model mist precies de dempers die zo'n spiraal in werkelijkheid afremmen — fiscale stabilisatoren (A0.3), reinstatement/omscholing (A0.8, C1.5) en augmentatie (A0.12).
- Daarom: **de vroege rem is de robuuste helft van de conclusie**, de late persistentie is kwalitatief geloofwaardig maar kwantitatief onzeker.

### Verwachte vervolgvragen

- *"Waarom trekt de rem later niet opnieuw aan als de ketenregeling nieuwe vaste contracten produceert?"* → Diensttijd wordt afgetopt door separaties (stabiliseert op 9,5 jaar, Figuur 34), de vergoeding per taak daalt en vlakt af (Figuur 33), en bij een AI-prijs op de vloer kantelt €28 extra geen beslissing meer. De data ondersteunen het sterk-zwak-sterk-verhaal simpelweg niet.
- *"Is de late divergentie geen artefact van één run?"* → Gemiddeld over 50 gepaarde seeds, met simultane band; wel eerlijk: heavy-tailed, de spreiding groeit aan het einde (dat zeg ik zelf in §7.1).
- *"Waarom sluit de poort niet ook in de baseline?"* → Wel eens, maar veel minder (Figuur 35, vlakke lijn): de hogere loonsom houdt de vraag en dus de marge op peil. Het verschil ís het mechanisme.
- *"Experiment 7 zegt dat de vergoeding het hele effect draagt — hoe rijmt dat?"* → Precies consistent: severance op 0 zetten reproduceert bijna het hele aan/uit-verschil (dz ≈ 0,8), dus het zaadje van fase 1 is de vergoeding zelf, niet de ketenregeling of de contractmix. De versterking (fase 3) is daarna generiek.

---

## Deel 2 — Waarom de skill-wage premie in élk scenario daalt (1,8 → 1,3)

### Het te verklaren feit

De premie w_h/w_l daalt in alle acht experimenten van ~1,8 naar ~1,3 — óók als er vrijwel niets wordt geautomatiseerd en óók met automatisering volledig uit. **De oorzaak is dus niet AI.** Het is een ingebouwde aanpassing van het startpunt, gedreven door de looncalibratie en de productiviteitsinstellingen (§7.2.5).

### De keten, stap voor stap

**Stap 1 — Bij de allocatie zijn de lonen bijna gelijk.** Op het allereerste moment staan de lonen op hun ondergrenzen: laaggeschoold op de minimumloonvloer **€14,71**, hooggeschoold op zijn looncurve-intercept **€15** (a_h). Verschil: 29 cent.

**Stap 2 — De warme start sorteert op comparatief voordeel.** Bij vrijwel gelijke lonen is de hooggeschoolde op bijna elke non-routine taak de goedkopere keuze (hogere productiviteit: exp(a·x) tegen exp(ξ·a·x) met ξ = 0,5, dus de laaggeschoolde produceert daar minder). De warme start zet dus de hooggeschoolden op het non-routine werk en de laaggeschoolden op het routinewerk — in de baseline **185 en 215 werkenden**.

**Stap 3 — Dán pas worden de lonen op die allocatie gezet, en die schieten uit elkaar.** `update_wages(instant=True)` zet de lonen op het niveau dat de allocatie impliceert via de looncurve: w_h = 15 + 0,10 × 185 ≈ **€33**, w_l = max(14,71; 6 + 0,06 × 215) ≈ **€19**. Startpremie ≈ 1,8.

**Stap 4 — De kostenvergelijking klapt om.** Bij €33 tegen €19 is de hooggeschoolde alleen nog de goedkopere optie op de *meest complexe* non-routine taken (daar is zijn productiviteitsvoorsprong groot genoeg om het loonverschil te compenseren). Voor de rest is de laaggeschoolde per eenheid output goedkoper geworden. In de baseline geldt dat voor **174 van de 185** banen die de hooggeschoolden bezetten: bijna hun hele startpositie is opeens "te duur".

**Stap 5 — Gewone verloop repareert het, vacature voor vacature.** Er is geen ontslaggolf; het gaat via exogene separaties (δ = 1,3%/maand). Elke keer dat een hooggeschoolde vertrekt, wordt de vrijgekomen taak via de matching (ronde 2, cross-skill met tolerantie θ = 1,2) gevuld met de nu goedkopere laaggeschoolde. De hooggeschoolde werkgelegenheid daalt dus geleidelijk.

**Stap 6 — De looncurve trekt w_h mee omlaag.** Omdat w_h via de curve aan L_h vastzit, glijdt het hooggeschoolde loon van ~€33 naar ~€24, terwijl w_l rond ~€19 blijft hangen (vlakkere curve, en de vloer). Een dalende teller boven een stabiele noemer = een dalende premie: 1,8 → 1,3.

**Stap 7 — Het proces stopt vanzelf.** Hoe lager w_h zakt, hoe meer taken waarop hooggeschoolden weer concurrerend zijn; hun werkgelegenheid vlakt af en de premie stabiliseert rond **1,3**. Het is een gedempte aanpassing naar het interne evenwicht van het model, geen instorting.

### Waarom dit de resultaten níet ondermijnt (het cruciale verweer)

1. **Common-mode: het valt weg in elke vergelijking.** Elk scenario start vanaf hetzélfde punt met dezélfde calibratie en dezélfde seeds. De gedeelde daling zit dus in beide curves van elke vergelijking en **valt weg in het gepaarde verschil**. Wat de dz-waarden in hoofdstuk 6 meten, is uitsluitend hoeveel méér of mínder de premie samendrukt onder een ander AI-kostenpad, comparatief voordeel of institutie.
2. **De premie wordt altijd relatief gelezen, nooit als niveau.** Nergens in de scriptie wordt "1,3" als voorspelling voor Nederland gepresenteerd.
3. **De wortel is benoemd:** het is een gevolg van twee vaste vaardigheidsgroepen in één sector, een steile hooggeschoolde looncurve en bijna-gelijke startlonen — allemaal expliciet in §7.2.5 en §7.4.

### De éénzinssamenvatting (memoriseren)

> "De daling van de premie is de aanpassing van het model aan zijn eigen startpunt: de warme start alloceert op bijna-gelijke lonen, de looncurve zet de lonen daarna ver uit elkaar, waardoor hooggeschoolden zichzelf uit de markt prijzen op alles behalve het complexste werk; verloop herverdeelt die taken en de looncurve trekt w_h mee omlaag tot een nieuw evenwicht rond 1,3. Het staat los van AI, is in §7.2.5 gediagnosticeerd, en valt weg in elke gepaarde vergelijking — dus de scenario-effecten op de premie blijven zuiver."

### Verwachte vervolgvragen

- *"Is dit geen bug?"* → Nee, het is de logische consequentie van twee calibratie-ankers die onderling niet consistent zijn gegeven de productiviteitsinstellingen: de CBS-loonniveaus (~€35/€21) én een gesegregeerde startallocatie. Het model lost die spanning endogeen op. Ik heb het gediagnosticeerd (§7.2.5) en het design (gepaarde seeds, verschilmeting) neutraliseert het.
- *"Waarom niet starten in het interne evenwicht (premie 1,3)?"* → Dan geef ik het empirische loonanker op (CBS-niveaus bij volledige werkgelegenheid) en verplaats ik het probleem naar de calibratie. De keuze is: herkenbaar Nederlands startpunt + gedocumenteerde transient, of een intern consistent maar empirisch losgezongen startpunt. Ik koos het eerste en meet alles als verschil.
- *"Zegt de daling iets over de werkelijkheid (dalende skill premium)?"* → Nee, en dat claim ik ook nergens: het niveau en de drift zijn een eigenschap van initialisatie en calibratie, geen voorspelling. Empirisch onderbouwde uitspraken over de premie vergen de rijkere skill-taakstructuur van §7.6.
- *"Hoe verhoudt dit zich tot de vroege premiedaling in Experiment 3?"* → Belangrijk onderscheid (§7.2.4): deze reallocatie-transient loopt via de looncurve en gewoon verloop, **niet** via de transitievergoeding, en is daardoor vrijwel identiek mét en zonder bescherming — hij valt weg in de gepaarde vergelijking en draagt niet bij aan de Experiment 3-kloof.

---

## Spiekbriefje: de twee ketens naast elkaar

| | Experiment 3 (late divergentie) | Premiedaling (overal) |
|---|---|---|
| Trigger | Transitievergoeding als NPV-wig op bezette taken | Bijna-gelijke startlonen bij allocatie |
| Motor | Keynesiaanse lus: Y↓ → A↓ → p↓ → poort dicht → geen heraanname | Looncurve: L_h↓ → w_h↓ terwijl w_l op vloer blijft |
| Tijdprofiel | vroeg significant → vlak midden → laat weer open | monotone daling, stabiliseert rond 1,3 |
| Draait om AI? | Ja (automatiseringsverschil is het zaadje) | Nee (gebeurt ook zonder automatisering) |
| Sleuteldiagnose | Fig. 33–37: rem zakt weg, poortblokkades/vraag/laaggeschoolde werkgelegenheid divergeren | Daling ook bij automatisering uit; 174/185 banen "te duur" na loonzetting |
| Kanttekening | Late gat = bovengrens, patrooncalibratie, heavy-tailed | Niveau ≠ voorspelling; valt weg in gepaarde verschillen |
| Kernzin | "Klein vroeg zaadje, door de vraaglus versterkt tot blijvend niveauverschil." | "Aanpassing aan het eigen startpunt; common-mode, dus zuivere scenario-effecten." |
