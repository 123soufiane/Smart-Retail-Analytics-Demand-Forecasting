"""Illustrative, human-readable labels for the anonymous Store Item dataset."""

import re

PRODUCT_NAMES = {
    f"item_{index}": name
    for index, name in enumerate(
        (
            "Whole Milk",
            "Sandwich Bread",
            "Fresh Bananas",
            "Bottled Water",
            "Large Eggs",
            "Ground Coffee",
            "Pasta",
            "Jasmine Rice",
            "Cheddar Cheese",
            "Plain Yogurt",
            "Orange Juice",
            "Chicken Breast",
            "Laundry Detergent",
            "Paper Towels",
            "Toothpaste",
            "Shampoo",
            "Olive Oil",
            "Breakfast Cereal",
            "Canned Tomatoes",
            "Sparkling Water",
            "Fresh Apples",
            "Frozen Vegetables",
            "Sliced Turkey",
            "Salted Butter",
            "Dish Soap",
            "Potato Chips",
            "Dark Chocolate",
            "Baby Diapers",
            "Facial Tissues",
            "Granola Bars",
            "Frozen Pizza",
            "Ground Beef",
            "Dry Dog Food",
            "Hand Soap",
            "Trash Bags",
            "Breakfast Sausage",
            "Strawberry Jam",
            "Peanut Butter",
            "Green Tea",
            "Energy Drink",
            "Salad Mix",
            "Cheddar Crackers",
            "Tomato Sauce",
            "Body Lotion",
            "Aluminum Foil",
            "Canned Tuna",
            "Rolled Oats",
            "Vanilla Ice Cream",
            "LED Light Bulbs",
            "AA Batteries",
        ),
        start=1,
    )
}

STORE_NAMES = {
    f"store_{index}": name
    for index, name in enumerate(
        (
            "Riverside Market",
            "Oakwood Market",
            "Lakeside Market",
            "Maple Street Market",
            "Cedar Grove Market",
            "Hillcrest Market",
            "Harbor Point Market",
            "Meadowbrook Market",
            "Pine Valley Market",
            "Westfield Market",
            "Brookside Market",
            "Willow Creek Market",
            "Northgate Market",
            "Fairview Market",
            "Stonebridge Market",
            "Greenfield Market",
            "Parkview Market",
            "Canyon Ridge Market",
            "Silver Lake Market",
            "Summit Plaza Market",
            "Garden District Market",
            "Milltown Market",
            "Sunset Square Market",
            "Highland Park Market",
            "Eastwood Market",
            "Redwood Market",
            "Clearwater Market",
            "Springbrook Market",
            "Bayside Market",
            "Old Town Market",
            "Crestview Market",
            "Forest Hill Market",
            "Grand Avenue Market",
            "Southpoint Market",
            "Elm Street Market",
            "Brighton Market",
            "Kingston Market",
            "Union Square Market",
            "Fox Run Market",
            "Central Station Market",
            "Bluebird Market",
            "Cloverfield Market",
        ),
        start=1,
    )
}


def display_product_name(sku: object, product_name: object = "") -> str:
    """Return a demo label for a Store Item SKU, or preserve its source name."""
    sku_match = re.fullmatch(r"SITEM-(item_\d+)", str(sku))
    name_match = re.fullmatch(r"Item (item_\d+)", str(product_name))
    item_id = sku_match.group(1) if sku_match else (
        name_match.group(1) if name_match else None
    )
    return PRODUCT_NAMES.get(item_id, str(product_name))


def display_store_name(store_name: object) -> str:
    """Return a demo label for a Store Item store, or preserve its source name."""
    match = re.fullmatch(r"Store Item (store_\d+)", str(store_name))
    return STORE_NAMES.get(match.group(1), str(store_name)) if match else str(store_name)
