def rows(cursor):
    return [dict(r) for r in cursor.fetchall()]
 
 
def validate_order(data):
    if not data:
        return "JSON body chahiye"
    if data.get("side") not in ("buy", "sell"):
        return "side 'buy' ya 'sell' hona chahiye"
    try:
        if float(data["price"]) <= 0:
            return "price 0 se zyada hona chahiye"
        if int(data["quantity"]) <= 0:
            return "quantity 0 se zyada honi chahiye"
    except (KeyError, ValueError, TypeError):
        return "price aur quantity valid numbers hone chahiye"
    return None