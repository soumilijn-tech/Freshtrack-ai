"""Produce Storage Knowledge Base and Advisory Service."""
from typing import Dict, Any, List

PRODUCE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Apple": {
        "category": "Fruit",
        "optimal_location": "Refrigerator Crisper Drawer",
        "ideal_temp": "1°C - 4°C (34°F - 39°F)",
        "humidity": "High (90-95%)",
        "ethylene_status": "High Ethylene Producer — store separately from leafy greens, carrots, and potatoes.",
        "tips": [
            "Keep in the crisper drawer with the humidity slider set to high.",
            "Separate bruised apples immediately; 'one bad apple spoils the bunch' is biochemically true due to rapid ethylene release.",
            "Can stay on the countertop for up to 5 days if eating soon.",
            "For cut slices, brush with dilute lemon juice or cold salt water to inhibit polyphenol oxidase browning."
        ],
        "shelf_life_days": {"Fresh": 28, "Fair / Consume Soon": 4, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop": "5-7 days", "Refrigerator": "3-6 weeks", "Freezer": "8 months (sliced/cooked)"},
        "freeze_friendly": True,
        "freezing_instructions": "Peel, core, slice, and toss with lemon juice. Freeze flat on a tray before transferring to freezer bags for baking or smoothies."
    },
    "Banana": {
        "category": "Fruit",
        "optimal_location": "Countertop (Room Temperature)",
        "ideal_temp": "15°C - 20°C (59°F - 68°F)",
        "humidity": "Moderate (50-60%)",
        "ethylene_status": "High Ethylene Producer — wrap stem crown tightly to slow ripening.",
        "tips": [
            "Wrap the stem cluster (crown) in plastic wrap or aluminum foil to throttle ethylene gas release.",
            "Do not refrigerate unripe green bananas; cold arrests ripening enzymes permanently.",
            "Once fully speckled yellow, you may refrigerate to extend firmness; the peel will turn brown but fruit remains sweet for 4-5 days.",
            "Keep away from other fruits unless you actively want to accelerate their ripening."
        ],
        "shelf_life_days": {"Fresh": 6, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop": "4-7 days", "Refrigerator": "4-5 days (when ripe)", "Freezer": "3-6 months (peeled)"},
        "freeze_friendly": True,
        "freezing_instructions": "Peel ripe or overripe bananas, slice into coins or freeze whole in ziploc bags. Perfect for banana bread, nice-cream, and smoothies."
    },
    "Orange": {
        "category": "Fruit",
        "optimal_location": "Refrigerator Crisper Drawer",
        "ideal_temp": "4°C - 7°C (39°F - 45°F)",
        "humidity": "Moderate-High (85-90%)",
        "ethylene_status": "Low Ethylene Producer & Low Sensitivity.",
        "tips": [
            "Store in an open mesh bag or ventilated bin for good air circulation; trapped moisture fosters penicillium green mold.",
            "Keep on the countertop in a fruit bowl if you plan to eat within 5-7 days.",
            "Bring to room temperature before juicing to maximize juice yield."
        ],
        "shelf_life_days": {"Fresh": 21, "Fair / Consume Soon": 5, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop": "7 days", "Refrigerator": "3-4 weeks", "Freezer": "6 months (zest and juice)"},
        "freeze_friendly": True,
        "freezing_instructions": "Freeze freshly squeezed juice in ice cube trays. Grate peel into zest and freeze in small sealed jars."
    },
    "Tomato": {
        "category": "Vegetable",
        "optimal_location": "Countertop (Stem-down away from sunlight)",
        "ideal_temp": "13°C - 18°C (55°F - 65°F)",
        "humidity": "Moderate (60%)",
        "ethylene_status": "Moderate Ethylene Producer — keep away from leafy greens and cucumbers.",
        "tips": [
            "Never refrigerate fresh unripe tomatoes! Cold temperatures below 12°C permanently destroy flavor aroma compounds (hexenals) and create a mealy texture.",
            "Store stem-side down on a flat surface to prevent moisture loss and inhibit microbial entry through the stem scar.",
            "Only refrigerate tomatoes that are completely overripe and soft, and consume within 2 days."
        ],
        "shelf_life_days": {"Fresh": 8, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop": "5-8 days", "Refrigerator": "2-3 days (overripe only)", "Freezer": "8 months (cooked/sauce)"},
        "freeze_friendly": True,
        "freezing_instructions": "Puree or roast with olive oil into tomato sauce, or freeze whole raw tomatoes (skin slips right off when thawed)."
    },
    "Potato": {
        "category": "Vegetable",
        "optimal_location": "Cool, Dark, Well-Ventilated Pantry",
        "ideal_temp": "7°C - 10°C (45°F - 50°F)",
        "humidity": "High (85-90%) in dark space",
        "ethylene_status": "Ethylene Sensitive — keep strictly away from onions and apples.",
        "tips": [
            "Store in a breathable paper bag, burlap sack, or dark basket. Sunlight triggers chlorophyll and toxic solanine glycoalkaloid synthesis (green skin).",
            "NEVER refrigerate raw potatoes; cold converts starches into reducing sugars (cold-induced sweetening) which darken and form acrylamide upon high-heat frying.",
            "Keep separate from onions; onions release volatile gases and moisture that cause potatoes to sprout prematurely."
        ],
        "shelf_life_days": {"Fresh": 35, "Fair / Consume Soon": 7, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Cool Dark Pantry": "4-6 weeks", "Room Temp": "1-2 weeks", "Refrigerator": "Do not refrigerate"},
        "freeze_friendly": False,
        "freezing_instructions": "Do not freeze raw. Par-boil or roast first, then freeze."
    },
    "Onion": {
        "category": "Vegetable",
        "optimal_location": "Dry, Cool, Dark Pantry or Mesh Basket",
        "ideal_temp": "4°C - 10°C (40°F - 50°F)",
        "humidity": "Low-Moderate (65-70%)",
        "ethylene_status": "Low Ethylene, but strong volatile sulfur emitters.",
        "tips": [
            "Keep in a dry, ventilated basket, mesh hanging bag, or open bin.",
            "Never store whole dry onions in sealed plastic bags; trapped humidity prompts fungal rotting and neck rot.",
            "Do not store near potatoes.",
            "Once cut, store onion halves in an airtight glass container in the refrigerator and use within 5 days."
        ],
        "shelf_life_days": {"Fresh": 40, "Fair / Consume Soon": 8, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Cool Dry Pantry": "1-2 months", "Countertop": "2-4 weeks", "Refrigerator": "5-7 days (cut only)"},
        "freeze_friendly": True,
        "freezing_instructions": "Dice and freeze raw without blanching in portioned freezer bags for quick sauteing."
    },
    "Carrot": {
        "category": "Vegetable",
        "optimal_location": "Refrigerator Crisper Drawer (High Humidity)",
        "ideal_temp": "0°C - 3°C (32°F - 37°F)",
        "humidity": "Very High (95%)",
        "ethylene_status": "Highly Ethylene Sensitive — exposure to apples or bananas causes isocoumarin accumulation, turning carrots bitter.",
        "tips": [
            "Immediately cut off leafy green tops; leaves draw moisture and vitamins from the taproot, turning carrots limp.",
            "Pro-tip: Store trimmed peeled or unpeeled carrots submerged in a container of cold water in the fridge (change water every 3 days) for 4 weeks of crisp snap!",
            "Keep away from apples, pears, and bananas."
        ],
        "shelf_life_days": {"Fresh": 25, "Fair / Consume Soon": 5, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Refrigerator": "3-4 weeks", "In Water Jar": "4-5 weeks", "Freezer": "10-12 months (blanched)"},
        "freeze_friendly": True,
        "freezing_instructions": "Slice, blanch in boiling water for 2 minutes, plunge into ice bath, pat dry, and freeze."
    },
    "Bell Pepper": {
        "category": "Vegetable",
        "optimal_location": "Refrigerator Crisper Drawer",
        "ideal_temp": "7°C - 10°C (45°F - 50°F)",
        "humidity": "High (90-95%)",
        "ethylene_status": "Low Producer, Moderately Sensitive to Ethylene.",
        "tips": [
            "Store completely dry in the crisper drawer inside a breathable produce bag or perforated bag.",
            "Do not wash until ready to slice.",
            "Green peppers are less ripe and last slightly longer than red, orange, or yellow peppers.",
            "Cover cut peppers with dry paper towel to catch condensation."
        ],
        "shelf_life_days": {"Fresh": 14, "Fair / Consume Soon": 3, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop": "3-4 days", "Refrigerator": "10-14 days", "Freezer": "8 months (sliced raw)"},
        "freeze_friendly": True,
        "freezing_instructions": "Stem, seed, slice into strips, and freeze raw on a sheet tray before packing into freezer containers. Great for stir fries and fajitas."
    },
    "Broccoli": {
        "category": "Vegetable",
        "optimal_location": "Refrigerator Crisper Drawer",
        "ideal_temp": "0°C - 2°C (32°F - 35°F)",
        "humidity": "Very High (95%)",
        "ethylene_status": "Extremely Ethylene Sensitive — yellowing florets indicate ethylene exposure or age.",
        "tips": [
            "Wrap loosely in damp paper towels, or stand the stem in a glass with 1 inch of water in the fridge like a bouquet of flowers.",
            "Do not wash before storing; moisture trapped in tight florets induces bacterial soft rot.",
            "Keep far away from apples and bananas."
        ],
        "shelf_life_days": {"Fresh": 9, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Refrigerator": "7-10 days", "Freezer": "10-12 months (blanched)"},
        "freeze_friendly": True,
        "freezing_instructions": "Cut into florets, blanch in boiling water for 3 minutes, shock in ice water, dry thoroughly, and freeze."
    },
    "Strawberry": {
        "category": "Fruit",
        "optimal_location": "Refrigerator Top Shelf / Main Compartment",
        "ideal_temp": "0°C - 2°C (32°F - 36°F)",
        "humidity": "High (90-95%)",
        "ethylene_status": "Low Ethylene Producer & Low Sensitivity.",
        "tips": [
            "Highly perishable; do not rinse or remove green hulls until moments before eating.",
            "Transfer from grocery carton into a shallow glass container lined with a dry paper towel in a single layer.",
            "Discard any moldy strawberry immediately to protect adjacent berries from botrytis cinerea grey mold.",
            "Optional hot water rinse (45°C / 113°F for 30 seconds) followed by thorough drying can destroy surface mold spores."
        ],
        "shelf_life_days": {"Fresh": 6, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Room Temp": "1-2 days", "Refrigerator": "5-7 days", "Freezer": "10 months"},
        "freeze_friendly": True,
        "freezing_instructions": "Hull, freeze whole or halved on parchment paper, then transfer into airtight freezer bags for smoothies and jams."
    },
    "Cucumber": {
        "category": "Vegetable",
        "optimal_location": "Refrigerator Door or Warmest Shelf",
        "ideal_temp": "10°C - 13°C (50°F - 55°F)",
        "humidity": "High (95%)",
        "ethylene_status": "Very Ethylene Sensitive — turns yellow and liquefies near bananas, melons, or tomatoes.",
        "tips": [
            "Cucumbers suffer chilling injury below 8°C (pitting, water-soaked flesh). Store in the refrigerator door or upper shelf, not the coldest back corners.",
            "Wrap each cucumber individually in a dry paper towel and place inside an open ziploc bag.",
            "Store away from tomatoes and apples."
        ],
        "shelf_life_days": {"Fresh": 9, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Cool Countertop": "3-5 days", "Refrigerator": "7-10 days", "Freezer": "Not recommended (high water)"},
        "freeze_friendly": False,
        "freezing_instructions": "Cucumbers turn mushy when frozen raw due to 95% water content. Pickle them instead or puree into iced gazpacho blocks."
    },
    "Lettuce": {
        "category": "Vegetable",
        "optimal_location": "Refrigerator Crisper Drawer (High Humidity)",
        "ideal_temp": "1°C - 4°C (34°F - 39°F)",
        "humidity": "Very High (95-98%)",
        "ethylene_status": "Extremely Ethylene Sensitive — develops rust-colored russet spotting when exposed to ethylene.",
        "tips": [
            "Wash, spin thoroughly dry in a salad spinner. Residual surface water causes rapid slime and cell collapse.",
            "Store leaves inside a sealed container layered with clean, dry paper towels to absorb transpiration moisture.",
            "Never store in the same drawer as apples, pears, or ripening avocados."
        ],
        "shelf_life_days": {"Fresh": 10, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Refrigerator": "7-12 days (properly dried)", "Countertop": "1 day", "Freezer": "Do not freeze"},
        "freeze_friendly": False,
        "freezing_instructions": "Freezing destroys leafy cellular structure, resulting in limp mush. Use fresh."
    },
    "Avocado": {
        "category": "Fruit",
        "optimal_location": "Countertop until ripe, then Refrigerator",
        "ideal_temp": "Room Temp (Ripening) -> 4°C (Storage)",
        "humidity": "Moderate (50-60%)",
        "ethylene_status": "High Ethylene Producer once ripening starts.",
        "tips": [
            "Ripen on the countertop. To speed ripening, place in a brown paper bag with a banana.",
            "Once the skin turns dark and gives slightly under gentle palm pressure, move to the refrigerator to arrest over-ripening for up to 5 days.",
            "For half an avocado: keep the pit in place, brush exposed flesh with olive oil or lime juice, and press plastic wrap directly onto the cut surface."
        ],
        "shelf_life_days": {"Fresh": 7, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop (Unripe)": "4-6 days", "Refrigerator (Ripe)": "4-5 days", "Freezer": "4 months (mashed)"},
        "freeze_friendly": True,
        "freezing_instructions": "Mash flesh with 1 tsp lemon or lime juice to prevent browning, pack into freezer bags with air pressed out."
    },
    "Lemon": {
        "category": "Fruit",
        "optimal_location": "Refrigerator Crisper Drawer",
        "ideal_temp": "4°C - 9°C (40°F - 48°F)",
        "humidity": "Moderate-High (85%)",
        "ethylene_status": "Low Ethylene Producer.",
        "tips": [
            "Lemons stored in a sealed ziploc bag in the refrigerator stay juicy and firm for up to 4 weeks (compared to 1 week dry on the counter).",
            "Microwave for 10-15 seconds before juicing to release double the juice.",
            "Zest organic lemons before juicing and store the zest in the freezer."
        ],
        "shelf_life_days": {"Fresh": 28, "Fair / Consume Soon": 5, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop": "7-10 days", "Refrigerator": "3-4 weeks", "Freezer": "6 months (zest and juice)"},
        "freeze_friendly": True,
        "freezing_instructions": "Freeze lemon juice in ice cube trays. Slice whole lemons into wheels and freeze on parchment paper for tea and cocktails."
    },
    "Persimmon": {
        "category": "Fruit",
        "optimal_location": "Countertop until soft/ripe, then Refrigerator Crisper Drawer",
        "ideal_temp": "15°C - 20°C (Ripening) -> 1°C - 4°C (Storage)",
        "humidity": "High (85-90%)",
        "ethylene_status": "Ethylene Sensitive — responds rapidly to ethylene to accelerate ripening and tannin reduction.",
        "tips": [
            "Distinguish your variety: Non-astringent Fuyu persimmons (squat, tomato-like) can be eaten while firm and crisp. Astringent Hachiya (acorn-shaped) must soften completely to a translucent jelly consistency before eating to eliminate puckering soluble tannins.",
            "Ripen firm astringent fruit stem-down on the countertop at room temperature away from direct sunlight. To accelerate ripening, enclose in a paper bag with an ethylene-producing apple or banana.",
            "Once ripe, transfer to the refrigerator crisper drawer to extend freshness for 2-3 weeks.",
            "Keep dry and check under the persistent 4-lobed calyx, as trapped moisture promotes surface fungal rot."
        ],
        "shelf_life_days": {"Fresh": 14, "Fair / Consume Soon": 4, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop (Firm)": "4-7 days", "Refrigerator (Ripe)": "2-3 weeks", "Freezer": "6-8 months (pureed or whole)"},
        "freeze_friendly": True,
        "freezing_instructions": "Freeze whole ripe Hachiya persimmons; slice off the calyx and scoop out flesh with a spoon as natural sorbet. Alternatively, peel and puree ripe flesh with a dash of lemon juice and freeze in sealed containers."
    },
    "Mango": {
        "category": "Fruit",
        "optimal_location": "Countertop until ripe, then Refrigerator",
        "ideal_temp": "18°C - 22°C (Ripening) -> 10°C - 12°C (Storage)",
        "humidity": "Moderate (85%)",
        "ethylene_status": "Moderate Ethylene Producer & Sensitive.",
        "tips": [
            "Never refrigerate hard, unripe mangoes; cold inhibits enzymatic ripening and dulls sweet flavor development.",
            "Ripen at room temperature until fruit yields slightly to gentle pressure and exudes a sweet tropical aroma.",
            "Move fully ripe mangoes to the refrigerator for up to 5 days."
        ],
        "shelf_life_days": {"Fresh": 7, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Countertop (Unripe)": "3-6 days", "Refrigerator (Ripe)": "5-7 days", "Freezer": "10 months (cubed)"},
        "freeze_friendly": True,
        "freezing_instructions": "Peel, dice into bite-size cubes, freeze in a single layer on parchment paper, then pack into freezer bags."
    },
    "Eggplant": {
        "category": "Vegetable",
        "optimal_location": "Cool Pantry or Upper Refrigerator Shelf",
        "ideal_temp": "10°C - 12°C (50°F - 54°F)",
        "humidity": "High (90%)",
        "ethylene_status": "Ethylene Sensitive — keep separate from apples, melons, and tomatoes.",
        "tips": [
            "Eggplants are chilling sensitive; temperatures below 7°C cause internal browning, pitting, and decay.",
            "Store in a breathable paper bag or wrap loosely in paper towels inside the crisper drawer.",
            "Use within 4-5 days of purchase while skin remains taut and glossy."
        ],
        "shelf_life_days": {"Fresh": 6, "Fair / Consume Soon": 2, "Overripe / Spoiled": 1},
        "shelf_life_table": {"Cool Pantry": "3-4 days", "Refrigerator": "5-7 days", "Freezer": "6 months (cooked/roasted)"},
        "freeze_friendly": True,
        "freezing_instructions": "Do not freeze raw. Roast or steam slices/puree first, then freeze."
    }
}

class StorageAdvisor:
    """Service providing storage guidance and shelf-life estimates."""

    @staticmethod
    def get_recommendation(produce_name: str, freshness_state: str = "Fresh") -> Dict[str, Any]:
        """Lookup storage recommendation for a given produce and freshness state."""
        # Clean and match produce name
        clean_name = produce_name.strip().title()
        
        # Match from knowledge base
        item_data = PRODUCE_KNOWLEDGE_BASE.get(clean_name)
        if not item_data:
            # Fallback for unlisted produce
            return {
                "location": "Refrigerator Crisper Drawer",
                "ideal_temp": "4°C (39°F)",
                "humidity": "Moderate",
                "ethylene_status": "Keep separated from strong ethylene producers (apples, bananas).",
                "tips": [
                    "Keep in a cool, well-ventilated storage container.",
                    "Inspect regularly and remove spoiling portions.",
                    "Wash only prior to consumption to preserve natural protective skin."
                ],
                "estimated_shelf_life_days": 5 if freshness_state == "Fresh" else (2 if "Fair" in freshness_state else 1),
                "freeze_friendly": True,
                "freezing_instructions": "Chop and freeze in portioned freezer bags for cooking."
            }

        shelf_life_days = item_data["shelf_life_days"].get(
            freshness_state, 
            item_data["shelf_life_days"]["Fresh"]
        )

        return {
            "location": item_data["optimal_location"],
            "ideal_temp": item_data["ideal_temp"],
            "humidity": item_data["humidity"],
            "ethylene_status": item_data["ethylene_status"],
            "tips": item_data["tips"],
            "estimated_shelf_life_days": shelf_life_days,
            "freeze_friendly": item_data["freeze_friendly"],
            "freezing_instructions": item_data["freezing_instructions"],
        }

    @staticmethod
    def get_all_guides() -> List[Dict[str, Any]]:
        """Return the complete produce knowledge base catalog."""
        results = []
        for name, data in PRODUCE_KNOWLEDGE_BASE.items():
            results.append({
                "name": name,
                "category": data["category"],
                "optimal_location": data["optimal_location"],
                "temperature_range": data["ideal_temp"],
                "humidity_recommendation": data["humidity"],
                "ethylene_profile": data["ethylene_status"],
                "preservation_tips": data["tips"],
                "shelf_life": data["shelf_life_table"],
                "can_freeze": data["freeze_friendly"],
                "freezing_method": data["freezing_instructions"]
            })
        return sorted(results, key=lambda x: x["name"])
