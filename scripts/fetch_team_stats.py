def load_cfbrank(path: str) -> dict:
    """
    Returns {team_name: rank_int} for the most recent season.
    Supports shapes like:
      A) {"2025":{"1":"Georgia","2":"Ohio State",...}}
      B) {"2025":[{"rank":1,"team":"Georgia"}, ...]}
      C) {"season":2025,"ranks":{"1":"Georgia",...}}
      D) {"season":2025,"ranks":[{"rank":1,"team":"Georgia"}, ...]}
      E) [{"season":2025,"ranks":{...}}, {"season":2024,"ranks":[...]}]
      F) [{"season":2025,"rankings":[{"rank":1,"team":"Georgia"}, ...]}, ...]
    If we can’t figure it out, returns {} (and we just won’t show model_rank).
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[warn] couldn't parse {path}: {e}")
        return {}

    def from_rank_dict(d: dict) -> dict:
        out = {}
        for rk, tm in d.items():
            try:
                out[str(tm)] = int(rk)
            except Exception:
                continue
        return out

    def from_rank_list(lst: list) -> dict:
        """
        Accept list of objects like:
          [{"rank":1,"team":"Georgia"}, {"rank":2,"team":"Ohio State"}, ...]
        Also tolerates {"Team":..,"Rank":..} or {"name":..}
        """
        out = {}
        for row in lst:
            if not isinstance(row, dict):
                # if it’s bare strings, there’s no rank info; skip
                continue
            team = row.get("team") or row.get("Team") or row.get("name")
            rk = row.get("rank") or row.get("Rank") or row.get("r") or row.get("R")
            try:
                rk = int(rk)
            except Exception:
                rk = None
            if team and rk:
                out[str(team)] = rk
        return out

    # ---- Top-level dict keyed by season (A/B)
    if isinstance(data, dict):
        # Seasons like "2025": ...
        seasons = [int(k) for k in data.keys() if str(k).isdigit()]
        if seasons:
            latest = str(max(seasons))
            payload = data.get(latest)

            if isinstance(payload, dict):
                # A) {"1":"Georgia", ...}
                return from_rank_dict(payload)

            if isinstance(payload, list):
                # B) [{"rank":1,"team":"Georgia"}, ...]
                return from_rank_list(payload)

        # C/D: {"season":2025,"ranks":{...}} or ranks:[...]
        if "ranks" in data:
            ranks = data["ranks"]
            if isinstance(ranks, dict):
                return from_rank_dict(ranks)
            if isinstance(ranks, list):
                return from_rank_list(ranks)

        # F (variant): {"season":2025,"rankings":[...]}
        if "rankings" in data and isinstance(data["rankings"], list):
            return from_rank_list(data["rankings"])

        return {}

    # ---- Top-level list of season objects (E/F)
    if isinstance(data, list) and data:
        # pick latest season that has usable ranks/rankings
        latest_obj = None
        latest_season = -10**9
        for obj in data:
            if not isinstance(obj, dict):
                continue
            try:
                s = int(obj.get("season"))
            except Exception:
                continue
            if s > latest_season:
                latest_season = s
                latest_obj = obj

        if latest_obj:
            if "ranks" in latest_obj:
                ranks = latest_obj["ranks"]
                if isinstance(ranks, dict):
                    return from_rank_dict(ranks)
                if isinstance(ranks, list):
                    return from_rank_list(ranks)
            if "rankings" in latest_obj and isinstance(latest_obj["rankings"], list):
                return from_rank_list(latest_obj["rankings"])

    return {}
