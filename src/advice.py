"""
advice.py — what the app tells the user *after* the model has spoken.

Two-layer design, on purpose:

  Layer 1 (default, always works):  an offline knowledge base with real
      agronomic content for all 38 PlantVillage classes. No internet, no API key,
      no rate limit, instant. Visitors being served in 8 seconds matters more
      during an exhibition than a fancy paragraph from a cloud LLM.

  Layer 2 (optional, opt-in):  the original project's idea — send the predicted
      class to an LLM (Mistral-7B via the Hugging Face Inference API) for extra
      narrative advice. If the token is missing or the Wi-Fi is down the app
      degrades gracefully instead of crashing.

That separation is also a good viva answer: "we made the demo dependency-free and
kept the LLM as an enhancement, not a single point of failure."
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Pretty names
# ---------------------------------------------------------------------------
CROP_LABELS = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn (maize)",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper,_bell": "Bell pepper",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}

# Dataset folder names that read badly when you just replace underscores.
DISEASE_LABELS = {
    "Apple___Apple_scab": "Apple scab",
    "Apple___Cedar_apple_rust": "Cedar apple rust",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Cercospora leaf spot (gray leaf spot)",
    "Corn_(maize)___Common_rust_": "Common rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern leaf blight",
    "Grape___Esca_(Black_Measles)": "Esca (black measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf blight (Isariopsis leaf spot)",
    "Orange___Haunglongbing_(Citrus_greening)": "Huanglongbing (citrus greening)",
    "Strawberry___Leaf_scorch": "Leaf scorch",
    "Tomato___Leaf_Mold": "Leaf mold",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Two-spotted spider mite",
    "Tomato___Target_Spot": "Target spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato yellow leaf curl virus",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus",
}


def parse_class_name(class_name: str) -> tuple[str, str]:
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""
    crop, _, condition = class_name.partition("___")
    return crop, condition


def crop_display(class_name: str) -> str:
    crop, _ = parse_class_name(class_name)
    return CROP_LABELS.get(crop, crop.replace("_", " "))


def disease_display(class_name: str) -> str:
    if class_name in DISEASE_LABELS:
        return DISEASE_LABELS[class_name]
    _, condition = parse_class_name(class_name)
    condition = condition.replace("_", " ").strip()
    if not condition:
        return "Unknown condition"
    return condition[0].upper() + condition[1:]


def human_readable(class_name: str) -> str:
    """'Grape___Black_rot' -> 'Grape — Black rot'"""
    return f"{crop_display(class_name)} — {disease_display(class_name)}"


def is_healthy(class_name: str) -> bool:
    return parse_class_name(class_name)[1].strip().lower() == "healthy"


# ---------------------------------------------------------------------------
# Knowledge base (38 entries — one per class)
# ---------------------------------------------------------------------------
KB: dict[str, dict[str, Any]] = {
    # ---------------------------- APPLE -----------------------------------
    "Apple___Apple_scab": dict(
        pathogen="Venturia inaequalis (fungus)",
        severity="Moderate to high",
        symptoms=[
            "Olive-green to black velvety spots on leaves, often near the veins",
            "Spots turn brown, leaves yellow and drop early",
            "Dark, scabby, cracked patches on the fruit surface",
        ],
        treatment=[
            "Remove and destroy fallen leaves in autumn — that is where the fungus overwinters",
            "Apply captan or myclobutanil starting at bud break, repeat every 10–14 days while wet weather continues",
            "Prune to open the canopy so leaves dry quickly after rain",
        ],
        prevention=[
            "Choose scab-resistant cultivars (e.g. Liberty, Enterprise) for new plantings",
            "Avoid overhead irrigation; water at the base in the morning",
            "Keep a 60 cm weed-free ring around the trunk to reduce humidity",
        ],
    ),
    "Apple___Black_rot": dict(
        pathogen="Botryosphaeria obtusa (fungus)",
        severity="Moderate",
        symptoms=[
            "Small purple 'frogeye' spots on leaves with a light centre",
            "Brown rot spreading on fruit, ending in black mummified fruit",
            "Sunken, cracked cankers on branches and trunk",
        ],
        treatment=[
            "Prune out cankers 15 cm below the visible edge during dry weather; disinfect blades between cuts",
            "Pick and destroy mummified fruit and dead wood",
            "Spray captan or thiophanate-methyl from green tip through summer",
        ],
        prevention=[
            "Keep trees vigorous but not over-fertilised — excess nitrogen invites cankers",
            "Remove wild or abandoned apple trees nearby (they are a reservoir)",
            "Protect fruit from bird and hail wounds; wounds are the entry point",
        ],
    ),
    "Apple___Cedar_apple_rust": dict(
        pathogen="Gymnosporangium juniperi-virginianae (fungus, needs juniper as alternate host)",
        severity="Moderate",
        symptoms=[
            "Bright yellow-orange spots on the upper leaf surface in spring",
            "Tube-like spore horns (aecia) on the underside of the same spots",
            "Fruit and twigs may show raised orange lesions",
        ],
        treatment=[
            "Apply myclobutanil or triadimefon at bloom and repeat 2–3 times at 10-day intervals",
            "Remove galls from nearby junipers/cedars before spring rain",
            "Strip severely infected fruit so the tree keeps strength",
        ],
        prevention=[
            "Do not plant apples within ~1 km of junipers if you can avoid it",
            "Prefer rust-resistant cultivars (e.g. Redfree, Liberty)",
            "Improve airflow and avoid evening overhead watering",
        ],
    ),
    "Apple___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Uniform green colour, no lesions, no distortion, firm fruit"],
        treatment=[
            "No chemical treatment needed — keep monitoring weekly",
            "Continue the normal fertiliser schedule (avoid excess nitrogen)",
        ],
        prevention=[
            "Prune every winter for light and airflow; remove dead wood",
            "Clear fallen leaves and fruit from around the trunk",
            "Water deeply but infrequently at the base, not on the foliage",
        ],
    ),
    # --------------------------- BLUEBERRY --------------------------------
    "Blueberry___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Glossy green leaves with no spotting, upright canes, no dieback"],
        treatment=["No treatment needed; keep the soil pH in check"],
        prevention=[
            "Keep soil pH between 4.5 and 5.5 — blueberries are acid lovers",
            "Mulch with pine bark or sawdust to hold moisture and suppress weeds",
            "Prune out old canes every winter so light reaches the centre",
        ],
    ),
    # ---------------------------- CHERRY ----------------------------------
    "Cherry_(including_sour)___Powdery_mildew": dict(
        pathogen="Podosphaera clandestina (fungus)",
        severity="Moderate",
        symptoms=[
            "White, powdery fungal growth on young leaves and shoot tips",
            "Leaves curl upward and may become distorted",
            "New growth looks stunted and pale",
        ],
        treatment=[
            "Spray sulphur or myclobutanil at first sign, repeat every 10–14 days",
            "Remove and bin the most affected shoot tips",
            "Avoid excessive nitrogen, which produces the soft growth mildew loves",
        ],
        prevention=[
            "Prune for an open canopy and good air movement",
            "Water in the morning at the base only",
            "Plant in full sun, away from damp, shaded walls",
        ],
    ),
    "Cherry_(including_sour)___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Clean green leaves, no white coating, no curled tips"],
        treatment=["No treatment needed; continue normal care"],
        prevention=[
            "Annual winter pruning for light and airflow",
            "Net the tree against birds as fruit ripens",
            "Remove fallen leaves to break disease cycles",
        ],
    ),
    # ----------------------------- CORN -----------------------------------
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": dict(
        pathogen="Cercospora zeae-maydis (fungus)",
        severity="Moderate to high",
        symptoms=[
            "Long, narrow, rectangular grey-tan lesions running parallel to the veins",
            "Lesions first appear on lower leaves and move upward",
            "Heavy infection kills leaves before the grain fills",
        ],
        treatment=[
            "Apply a strobilurin (azoxystrobin) or propiconazole fungicide at tasselling if lesions are spreading",
            "Scout the field at the V8–VT stage to decide whether a spray is justified",
            "Destroy or incorporate infected residue after harvest",
        ],
        prevention=[
            "Rotate to a non-host crop (soybean) for at least one season",
            "Choose hybrids with a high gray-leaf-spot resistance rating",
            "Avoid continuous no-till maize on the same field",
        ],
    ),
    "Corn_(maize)___Common_rust_": dict(
        pathogen="Puccinia sorghi (fungus)",
        severity="Low to moderate",
        symptoms=[
            "Small, round, reddish-brown pustules on both leaf surfaces",
            "Pustules burst and release orange-brown powder on your finger",
            "Severe cases turn leaves yellow and dry early",
        ],
        treatment=[
            "Fungicide (propiconazole, pyraclostrobin) only if pustules appear before tasselling",
            "Usually not worth spraying after the grain-fill stage",
            "Note the infected field so you can rotate next season",
        ],
        prevention=[
            "Plant rust-resistant hybrids — the single most effective control",
            "Early planting escapes the peak rust pressure",
            "Avoid excessive nitrogen, which increases susceptibility",
        ],
    ),
    "Corn_(maize)___Northern_Leaf_Blight": dict(
        pathogen="Exserohilum turcicum (fungus)",
        severity="Moderate to high",
        symptoms=[
            "Long, cigar-shaped grey-green lesions up to 15 cm long",
            "Lesions start on lower leaves and spread upward",
            "Severe infection causes premature leaf death and poor ear fill",
        ],
        treatment=[
            "Fungicide at VT/R1 (tassel/silk) if lesions are already on the upper leaves",
            "Remove or bury crop residue after harvest",
            "Record the disease level for next season's hybrid choice",
        ],
        prevention=[
            "Rotate maize with a non-host crop and manage residue",
            "Use hybrids with Ht resistance genes where available",
            "Balanced potassium fertility reduces lesion severity",
        ],
    ),
    "Corn_(maize)___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Uniform green leaves, no lesions or pustules, no yellowing"],
        treatment=["No treatment needed; continue scouting every 7–10 days"],
        prevention=[
            "Rotate crops and manage residue to keep inoculum low",
            "Keep nitrogen and potassium balanced",
            "Avoid overhead irrigation late in the day",
        ],
    ),
    # ----------------------------- GRAPE ----------------------------------
    "Grape___Black_rot": dict(
        pathogen="Guignardia bidwellii (fungus)",
        severity="High",
        symptoms=[
            "Tan leaf spots with a dark brown border and tiny black dots (pycnidia)",
            "Berries shrivel into hard black mummies",
            "Small black lesions on young shoots and tendrils",
        ],
        treatment=[
            "Remove and destroy all mummified berries and infected canes",
            "Spray mancozeb or myclobutanil from bud break until fruit set, then rotate chemistry",
            "Open up the canopy so fruit dries quickly",
        ],
        prevention=[
            "Keep the fruit zone at least 1 m above the ground",
            "Remove wild grapes and Virginia creeper nearby",
            "Start the spray programme early — black rot is much easier to prevent than to stop",
        ],
    ),
    "Grape___Esca_(Black_Measles)": dict(
        pathogen="Phaeomoniella / Phaeoacremonium species (fungal complex in the wood)",
        severity="High",
        symptoms=[
            "'Tiger-stripe' leaves: yellow or red patches between the veins",
            "Sudden wilting of one shoot ('apoplexy') in mid-summer",
            "Dark streaking inside the wood; black measles spots on berries",
        ],
        treatment=[
            "There is no chemical cure — prune out affected wood down to clean, white tissue",
            "Apply a pruning wound paste to the fresh cuts",
            "Remove and replant vines that are already collapsing",
        ],
        prevention=[
            "Prune only in dry weather and disinfect blades between vines",
            "Keep vines free from drought and waterlogging stress",
            "Use certified disease-free planting material",
        ],
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": dict(
        pathogen="Pseudocercospora vitis (fungus)",
        severity="Moderate",
        symptoms=[
            "Irregular brown lesions on older leaves, often with a yellow halo",
            "Lesions merge and cause premature leaf drop",
            "Most visible late in the season after harvest",
        ],
        treatment=[
            "Apply copper or mancozeb if lesions are spreading after harvest",
            "Remove and bury fallen leaves",
            "Do not let the vine go into winter with a heavy infection",
        ],
        prevention=[
            "Canopy management for light and airflow",
            "Keep nitrogen moderate so leaves are not overly soft",
            "Collect and remove leaf litter in autumn",
        ],
    ),
    "Grape___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Flat, evenly green leaves, no lesions, healthy fruit set"],
        treatment=["No treatment needed; continue the regular protective programme"],
        prevention=[
            "Maintain canopy management (shoot thinning, leaf removal in the fruit zone)",
            "Monitor weekly from bud break — growers say 'the best spray is scouting'",
            "Keep a calendar record of sprays and weather events",
        ],
    ),
    # ----------------------------- ORANGE ---------------------------------
    "Orange___Haunglongbing_(Citrus_greening)": dict(
        pathogen="'Candidatus Liberibacter asiaticus' (bacterium, spread by the Asian citrus psyllid)",
        severity="Very high — no cure",
        symptoms=[
            "Blotchy, asymmetric yellow mottle on the leaves (yellow on one side of the midrib only)",
            "Lopsided, small, bitter fruit that stays green at the bottom",
            "Young shoots yellow, twigs die back progressively",
        ],
        treatment=[
            "There is no cure: remove and destroy confirmed infected trees to protect the rest",
            "Control the psyllid vector with imidacloprid or horticultural oil rotations",
            "Keep affected trees productive for as long as possible only if the infection is light, and replant with certified stock",
        ],
        prevention=[
            "Buy only certified disease-free nursery stock",
            "Install psyllid monitoring traps and inspect new growth monthly",
            "Do not move fruit, leaves or budwood from infected areas",
        ],
    ),
    # ----------------------------- PEACH ----------------------------------
    "Peach___Bacterial_spot": dict(
        pathogen="Xanthomonas arboricola pv. pruni (bacterium)",
        severity="Moderate",
        symptoms=[
            "Small, water-soaked, angular spots that turn purple-brown",
            "Centres of the spots drop out, giving a 'shot-hole' look",
            "Sunken, pitted lesions on the fruit",
        ],
        treatment=[
            "Spray copper or oxytetracycline in early spring and again after harvest",
            "Never spray copper during the heat of the day — it burns the leaves",
            "Remove the most severely affected branches",
        ],
        prevention=[
            "Plant resistant cultivars and give them full sun",
            "Put a windbreak on the windy side — wind-driven rain spreads the bacteria",
            "Avoid excessive nitrogen; keep the tree in moderate vigour",
        ],
    ),
    "Peach___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Even green leaves without holes or spots, smooth fruit"],
        treatment=["No treatment needed; keep the routine spray calendar"],
        prevention=[
            "Winter pruning for an open vase shape",
            "Thin fruit so branches are not overloaded",
            "Remove and destroy prunings and fallen fruit",
        ],
    ),
    # -------------------------- BELL PEPPER -------------------------------
    "Pepper,_bell___Bacterial_spot": dict(
        pathogen="Xanthomonas euvesicatoria (bacterium)",
        severity="Moderate to high",
        symptoms=[
            "Small water-soaked spots that become brown with a yellow halo",
            "Leaves drop, exposing fruit to sunburn",
            "Raised, scabby spots on the pepper surface",
        ],
        treatment=[
            "Spray a copper + mancozeb tank mix every 7–10 days in wet weather (copper alone is not enough)",
            "Remove and bin infected leaves and fruit — do not compost them",
            "Switch to drip irrigation immediately",
        ],
        prevention=[
            "Use certified pathogen-free seed and transplants",
            "Rotate away from peppers, tomatoes and aubergines for 2 years",
            "Mulch to stop soil splashing onto the lower leaves",
        ],
    ),
    "Pepper,_bell___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Glossy dark-green leaves, firm fruit, no spotting"],
        treatment=["No treatment needed; keep watering and feeding consistent"],
        prevention=[
            "Water at the base early in the day",
            "Stake plants so fruit does not touch the soil",
            "Rotate the bed every season",
        ],
    ),
    # ----------------------------- POTATO ---------------------------------
    "Potato___Early_blight": dict(
        pathogen="Alternaria solani (fungus)",
        severity="Moderate",
        symptoms=[
            "Brown 'target-board' lesions with concentric rings on older leaves",
            "A yellow halo around each lesion",
            "Lesions start low on the plant and move up",
        ],
        treatment=[
            "Spray chlorothalonil or mancozeb every 7–10 days once symptoms appear",
            "Remove the oldest affected leaves to slow the spread",
            "Keep plants adequately fed — stressed plants get hit hardest",
        ],
        prevention=[
            "Rotate with a non-solanaceous crop for 3 years",
            "Destroy volunteer potatoes and crop debris",
            "Irrigate in the morning at the base, never overhead in the evening",
        ],
    ),
    "Potato___Late_blight": dict(
        pathogen="Phytophthora infestans (oomycete) — the Irish Famine pathogen",
        severity="Very high — can destroy a field in days",
        symptoms=[
            "Water-soaked, greasy grey-green lesions that spread rapidly",
            "White fuzzy mould on the underside of the lesion in humid weather",
            "The whole plant collapses; tubers develop reddish-brown dry rot",
        ],
        treatment=[
            "Act within 24 hours: apply a preventive/curative product (metalaxyl + mancozeb, cymoxanil) and repeat in 7 days",
            "Cut and remove the haulm of badly infected plants; do not leave them in the field",
            "Harvest early if the epidemic is out of control and the tubers are mature enough",
        ],
        prevention=[
            "Always use certified seed potatoes — never plant supermarket tubers",
            "Destroy cull piles and volunteers before planting",
            "Watch the weather: two consecutive days of high humidity and 15–20 °C means spray now",
        ],
    ),
    "Potato___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Even green foliage, no lesions, no wilting"],
        treatment=["No treatment needed; continue the routine protectant programme in wet weather"],
        prevention=[
            "Rotate with cereals or legumes",
            "Earthing up to protect tubers from spores washing down",
            "Remove volunteer plants and cull piles",
        ],
    ),
    # --------------------------- RASPBERRY --------------------------------
    "Raspberry___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Green, unspotted leaves, strong new canes, clean fruit"],
        treatment=["No treatment needed"],
        prevention=[
            "Prune out floricanes after harvest and burn them",
            "Keep rows narrow and supported so air moves through",
            "Drip irrigation only — raspberry roots hate waterlogging",
        ],
    ),
    # ---------------------------- SOYBEAN ---------------------------------
    "Soybean___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Uniform green trifoliate leaves, no lesions, no yellowing"],
        treatment=["No treatment needed; scout weekly for insects and rust"],
        prevention=[
            "Rotate with maize or wheat",
            "Use a good inoculant at planting if the field is new to soybean",
            "Keep residue management in mind to reduce soil-borne inoculum",
        ],
    ),
    # ----------------------------- SQUASH ---------------------------------
    "Squash___Powdery_mildew": dict(
        pathogen="Podosphaera xanthii (fungus)",
        severity="Moderate",
        symptoms=[
            "White talcum-powder-like patches on the upper leaf surface",
            "Leaves yellow, dry and shrivel at the edges",
            "Plants decline early and fruit stays small",
        ],
        treatment=[
            "Spray sulphur, potassium bicarbonate or a horticultural oil at the first patch, repeat weekly",
            "Remove the worst-affected leaves and bin them",
            "Alternate products so resistance does not build up",
        ],
        prevention=[
            "Full sun and wide spacing — this fungus loves shade and still air",
            "Water in the morning at the base only",
            "Grow resistant or tolerant varieties where you can",
        ],
    ),
    # --------------------------- STRAWBERRY -------------------------------
    "Strawberry___Leaf_scorch": dict(
        pathogen="Diplocarpon earlianum (fungus)",
        severity="Moderate",
        symptoms=[
            "Small purple spots on the upper leaf surface that enlarge and darken",
            "Leaf margins and tips look scorched, as though burned",
            "Heavy infection leaves the plant weak going into winter",
        ],
        treatment=[
            "Remove and destroy infected leaves after harvest",
            "Spray captan or myclobutanil at renovation and again in early autumn",
            "Renovate the bed: mow the old foliage and thin the plants",
        ],
        prevention=[
            "Avoid overcrowded beds — 30 cm between plants",
            "Replace plantings every 2–3 years",
            "Use a straw mulch so spores cannot splash up from the soil",
        ],
    ),
    "Strawberry___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Glossy green leaves, no spots, firm red fruit"],
        treatment=["No treatment needed"],
        prevention=[
            "Renovate the bed after harvest and renew mulch",
            "Keep weeds down — they shelter both pests and inoculum",
            "Water at the base, early in the day",
        ],
    ),
    # ----------------------------- TOMATO ---------------------------------
    "Tomato___Bacterial_spot": dict(
        pathogen="Xanthomonas perforans (bacterium)",
        severity="Moderate",
        symptoms=[
            "Small, dark, water-soaked spots with a yellow halo",
            "Leaves yellow and drop, leaving fruit exposed to sunscald",
            "Scabby, raised spots on green fruit",
        ],
        treatment=[
            "Spray copper + mancozeb every 7 days during wet weather",
            "Remove symptomatic leaves; do not compost them",
            "Switch from sprinkler to drip irrigation straight away",
        ],
        prevention=[
            "Rotate out of tomatoes/peppers for at least 2 years",
            "Use certified clean seed (or hot-water-treat seed at 52 °C for 20 min)",
            "Mulch and stake so foliage stays off the soil",
        ],
    ),
    "Tomato___Early_blight": dict(
        pathogen="Alternaria linariae / A. solani (fungus)",
        severity="Moderate",
        symptoms=[
            "Brown lesions with concentric rings ('target board') on the lower leaves",
            "A yellow halo around each lesion",
            "Progressive defoliation from the bottom of the plant upward",
        ],
        treatment=[
            "Remove the affected lower leaves and bin them",
            "Spray chlorothalonil or copper-based fungicide every 7–10 days",
            "Mulch around the base to stop soil splash",
        ],
        prevention=[
            "Rotate the bed for 3 years",
            "Stake and prune to keep leaves off the ground",
            "Water at the base in the morning; keep the foliage dry",
        ],
    ),
    "Tomato___Late_blight": dict(
        pathogen="Phytophthora infestans (oomycete)",
        severity="Very high — destroys a crop in days",
        symptoms=[
            "Large, greasy, grey-green lesions with a pale border",
            "White fuzzy mould on the underside of the lesions in humid weather",
            "Brown firm rot on green fruit; the plant collapses quickly",
        ],
        treatment=[
            "Remove and destroy affected plants immediately (bag them — do not compost)",
            "Protect the remaining plants with a preventive spray (chlorothalonil, mancozeb, copper)",
            "Harvest mature green fruit early if the outbreak is advancing",
        ],
        prevention=[
            "Avoid overhead watering and never work with wet plants",
            "Space and stake plants so the canopy dries fast",
            "Grow resistant/tolerant varieties and avoid planting near potatoes",
        ],
    ),
    "Tomato___Leaf_Mold": dict(
        pathogen="Passalora fulva (fungus)",
        severity="Moderate",
        symptoms=[
            "Pale yellow patches on the upper leaf surface",
            "Olive-green to grey velvet mould on the underside, directly below the patch",
            "Leaves curl, wither and drop",
        ],
        treatment=[
            "Increase ventilation and reduce humidity immediately (it is a greenhouse disease)",
            "Remove affected leaves and bin them",
            "Apply copper or a biofungicide if conditions stay humid",
        ],
        prevention=[
            "Keep glasshouse humidity below ~85 % and ventilate daily",
            "Water in the morning, never in the evening",
            "Grow resistant cultivars where available",
        ],
    ),
    "Tomato___Septoria_leaf_spot": dict(
        pathogen="Septoria lycopersici (fungus)",
        severity="Moderate",
        symptoms=[
            "Small circular spots with grey-white centres and dark margins",
            "Tiny black dots (pycnidia) inside the spots — the diagnostic feature",
            "Lower leaves yellow, die and drop; yield drops",
        ],
        treatment=[
            "Remove infected lower leaves and destroy them",
            "Spray copper or mancozeb every 7–10 days",
            "Mulch to block soil splash, which is how it spreads",
        ],
        prevention=[
            "3-year rotation away from tomatoes and potatoes",
            "Stake and prune for a dry canopy",
            "Clean up all debris at the end of the season",
        ],
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": dict(
        pathogen="Tetranychus urticae (mite — a pest, not a disease)",
        severity="Moderate to high in dry heat",
        symptoms=[
            "Fine yellow stippling on the leaves, giving a bronzed, dusty look",
            "Delicate silk webbing under the leaves and at the growing tips",
            "Leaves dry out and drop; plants stall in hot, dry weather",
        ],
        treatment=[
            "Spray insecticidal soap or horticultural oil, covering the leaf undersides; repeat after 5 days",
            "Use a miticide (abamectin, bifenazate) for heavy infestations and rotate chemistry",
            "Raise humidity and mist the plants — mites hate moisture",
        ],
        prevention=[
            "Check the underside of leaves weekly during hot spells",
            "Avoid excess nitrogen and water stress",
            "Encourage natural predators (Phytoseiulus persimilis)",
        ],
    ),
    "Tomato___Target_Spot": dict(
        pathogen="Corynespora cassiicola (fungus)",
        severity="Moderate",
        symptoms=[
            "Brown lesions with concentric rings and a yellow halo",
            "Small pitted spots on the fruit",
            "Canopy thins from the bottom up",
        ],
        treatment=[
            "Apply azoxystrobin or chlorothalonil on a 7–10 day interval",
            "Remove infected foliage; improve airflow",
            "Avoid overhead irrigation",
        ],
        prevention=[
            "Rotate crops and manage residue",
            "Stake and prune for an open canopy",
            "Mulch to reduce splash dispersal",
        ],
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": dict(
        pathogen="Tomato yellow leaf curl virus (TYLCV) — a begomovirus spread by whitefly",
        severity="Very high — no cure",
        symptoms=[
            "Upward curling and yellowing of the leaflet margins on new growth",
            "Severely stunted plants with a bushy, bunched look",
            "Flowers drop; almost no fruit set",
        ],
        treatment=[
            "There is no cure: rogue out and destroy infected plants as soon as they are confirmed",
            "Control the whitefly vector (yellow sticky traps + insecticidal soap or a systemic insecticide)",
            "Do not keep 'probably infected' plants next to healthy ones",
        ],
        prevention=[
            "Use virus-resistant/tolerant hybrids — the only reliable defence",
            "Exclude whiteflies with a 50-mesh insect net in a tunnel or glasshouse",
            "Plant away from older infected crops and remove weeds that host whitefly",
        ],
    ),
    "Tomato___Tomato_mosaic_virus": dict(
        pathogen="Tomato mosaic virus (ToMV, a tobamovirus)",
        severity="High",
        symptoms=[
            "Light and dark green mosaic mottle on the leaves",
            "Leaf distortion, 'fern-leaf' narrowing, stunted growth",
            "Brown streaking inside the stem; fruit may show internal browning",
        ],
        treatment=[
            "No chemical cure: remove and destroy infected plants",
            "Wash hands and tools with soap or a 1 % trisodium phosphate solution between plants",
            "Do not smoke or handle tobacco near the crop — tobacco can carry the virus",
        ],
        prevention=[
            "Sow certified virus-free seed and use resistant cultivars with the Tm-2 gene",
            "Disinfect pruners, stakes and gloves between plants",
            "Control weeds and volunteers; keep the crop off the ground",
        ],
    ),
    "Tomato___healthy": dict(
        pathogen="None — healthy tissue",
        severity="None",
        symptoms=["Even green leaves, no mottling, no spotting, strong new growth"],
        treatment=["No treatment needed; keep watering and feeding consistent"],
        prevention=[
            "Stake and prune for airflow",
            "Water at the base, in the morning",
            "Rotate the bed and clean up end-of-season debris",
        ],
    ),
}

assert len(KB) == 38, f"knowledge base should cover all 38 classes, has {len(KB)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_advice(class_name: str) -> dict[str, Any]:
    """
    Full, UI-ready advice record for a predicted class.

    Always returns a dict — an unknown class falls back to a generic message so
    the demo can never crash on a typo.
    """
    entry = KB.get(class_name)
    healthy = is_healthy(class_name)
    base = {
        "class_name": class_name,
        "crop": crop_display(class_name),
        "disease": disease_display(class_name),
        "human": human_readable(class_name),
        "is_healthy": healthy,
    }
    if entry is None:
        base.update(
            pathogen="Unknown",
            severity="Unknown",
            symptoms=["No reference data for this class."],
            treatment=["Inspect the plant visually and consult a local agronomist."],
            prevention=["Send a clear photo of a single leaf to a plant clinic."],
        )
        return base
    base.update(entry)
    return base


def format_advice(class_name: str) -> str:
    """Plain-text rendering, used by predict_cli.py and the JSON report."""
    a = get_advice(class_name)
    lines = [
        f"{a['human']}",
        f"Pathogen : {a['pathogen']}",
        f"Severity : {a['severity']}",
        "",
        "Symptoms observed:",
        *[f"  - {s}" for s in a["symptoms"]],
        "",
        "Treatment:",
        *[f"  {i + 1}. {t}" for i, t in enumerate(a["treatment"])],
        "",
        "Prevention:",
        *[f"  - {p}" for p in a["prevention"]],
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional layer 2: LLM narrative (the original project's idea, made optional)
# ---------------------------------------------------------------------------
def llm_advice(class_name: str, token: str | None = None, timeout: float = 30.0) -> str | None:
    """
    Ask a hosted LLM for extra narrative advice.

    Returns None (never raises) if the token is missing, the package is not
    installed, or the network is unavailable — the app then simply shows the
    offline knowledge base instead.
    """
    token = token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        return None

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(provider="featherless-ai", api_key=token, timeout=timeout)
        a = get_advice(class_name)
        prompt = (
            f"You are an expert plant pathologist. The diagnosis is: {a['human']}. "
            f"Pathogen: {a['pathogen']}. Severity: {a['severity']}. "
            "Give 4 short, actionable care points for this exact condition. "
            "Be specific, no filler, no conditionals."
        )
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": "You are a concise plant health expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else None
    except Exception as exc:                     # noqa: BLE001 - optional feature
        print(f"[advice] LLM advice unavailable ({type(exc).__name__}: {exc})")
        return None
