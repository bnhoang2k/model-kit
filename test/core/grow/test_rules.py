import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.grow.rules import KIT_RULES, GrowthRules, RuleReader, load_rules

TINY = Path(__file__).parent / "fixtures" / "tiny.rules.json"
TINY_BUDS = Path(__file__).parent / "fixtures" / "tiny-buds.rules.json"
TUBE_SIDES_ROWS = ("tubes.sides", "lite.tree-line")


def tiny_data() -> dict:
    return json.loads(TINY.read_text())


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "x.rules.json"
    path.write_text(json.dumps(data))
    return path


def reader() -> RuleReader:
    return RuleReader(load_rules(TINY))


def test_loads_tiny():
    rules = load_rules(TINY)
    assert isinstance(rules, GrowthRules)
    assert rules.species == "tuftbranch"
    assert {"trunk.size", "bark.colour"} <= set(rules.rows)
    assert {"A", "B"} <= set(rules.sources)
    assert "bark.colour" in rules.notes
    assert rules.rows["crown.base"].level == "crown"
    assert rules.params["crown.base_of_height"].rows == ("crown.base",)
    r = RuleReader(rules)
    assert r.num("crown.base_of_height") == 0.42
    assert r.range("trunk.height_m") == (1.6, 2.4)
    assert r.int_range("branch.angle_deg") == (35, 55)
    assert r.choices("branch.count_choices") == (3.0, 4.0, 5.0)
    assert r.int_choices("branch.count_choices") == (3, 4, 5)
    assert r.int("branch.tiers") == 2
    assert r.names("leaf.kinds") == ("oval", "lobed")
    pts = r.points("leaf.outline")
    assert isinstance(pts, np.ndarray) and pts.shape == (4, 2)


def test_param_cites_unknown_row_raises(tmp_path):
    data = tiny_data()
    data["params"]["trunk.girth_m"]["rows"] = ["trunk.nope"]
    with pytest.raises(ManifestError, match=r"params\.trunk\.girth_m\.rows\[0\].*trunk\.nope"):
        load_rules(write(tmp_path, data))


def test_row_unused_raises(tmp_path):
    data = tiny_data()
    data["notes"] = []
    with pytest.raises(ManifestError, match=r"bark\.colour.*no param cites it"):
        load_rules(write(tmp_path, data))


def test_note_that_a_param_cites_raises(tmp_path):
    data = tiny_data()
    data["params"]["trunk.girth_m"]["rows"].append("bark.colour")
    with pytest.raises(ManifestError, match=r"notes\[0\].*bark\.colour"):
        load_rules(write(tmp_path, data))


def test_unknown_source_key_raises(tmp_path):
    data = tiny_data()
    data["rows"][0]["source"] = "field notes [XX]"
    with pytest.raises(ManifestError, match=r"rows\[0\]\.source.*'XX'"):
        load_rules(write(tmp_path, data))


def test_rule_text_is_free_and_slug_may_differ():
    rules = load_rules(TINY)
    row = rules.rows["leaf.shape"]
    assert row.rule == "tip / base shape"
    assert row.id == "leaf.shape"


@pytest.mark.parametrize("bad_id", ["trunk.base", "crown", "crown.", "crown.a b", "crown.a/b"])
def test_row_id_must_be_level_dot_slug(tmp_path, bad_id):
    data = tiny_data()
    data["rows"][1]["id"] = bad_id
    with pytest.raises(ManifestError, match=r"rows\[1\]\.id.*'crown'"):
        load_rules(write(tmp_path, data))


def test_row_id_slug_may_differ_from_rule_text(tmp_path):
    data = tiny_data()
    data["rows"][1].update(id="crown.tree-line", rule="tree line")
    data["params"]["crown.base_of_height"]["rows"] = ["crown.tree-line"]
    assert "crown.tree-line" in load_rules(write(tmp_path, data)).rows


@pytest.mark.parametrize("item", [["a"], {"a": 1}, 3, None])
def test_non_text_row_ref_raises_manifest_error(tmp_path, item):
    data = tiny_data()
    data["notes"] = ["bark.colour", item]
    with pytest.raises(ManifestError, match=r"notes\[1\]"):
        load_rules(write(tmp_path, data))
    data = tiny_data()
    data["params"]["trunk.girth_m"]["rows"] = ["trunk.size", item]
    with pytest.raises(ManifestError, match=r"params\.trunk\.girth_m\.rows\[1\]"):
        load_rules(write(tmp_path, data))


def test_repeated_row_id_raises(tmp_path):
    data = tiny_data()
    data["rows"][1] = copy.deepcopy(data["rows"][0])
    with pytest.raises(ManifestError, match="repeat"):
        load_rules(write(tmp_path, data))


@pytest.mark.parametrize("where", ["top", "row", "param", "source"])
def test_unknown_field_raises(tmp_path, where):
    data = tiny_data()
    target = {
        "top": data,
        "row": data["rows"][0],
        "param": data["params"]["trunk.girth_m"],
        "source": data["sources"][0],
    }[where]
    target["extra"] = 1
    with pytest.raises(ManifestError, match="unknown field 'extra'"):
        load_rules(write(tmp_path, data))


def test_missing_field_raises(tmp_path):
    data = tiny_data()
    del data["notes"]
    with pytest.raises(ManifestError, match="missing field 'notes'"):
        load_rules(write(tmp_path, data))


def test_repeated_key_raises(tmp_path):
    path = tmp_path / "x.rules.json"
    text = TINY.read_text().replace('"value": 0.22,', '"value": 0.22, "value": 0.3,')
    path.write_text(text)
    with pytest.raises(ManifestError, match="'value' repeats"):
        load_rules(path)


@pytest.mark.parametrize(
    "value",
    [True, "x", [], [1, "a"], [[1, 2, 3]], [[1, 2], 3], [1, [2, 3]], None, {"a": 1}],
)
def test_bad_param_value_raises(tmp_path, value):
    data = tiny_data()
    data["params"]["trunk.girth_m"]["value"] = value
    with pytest.raises(ManifestError, match=r"params\.trunk\.girth_m\.value"):
        load_rules(write(tmp_path, data))


def test_reader_records_keys():
    r = reader()
    assert r.read == frozenset()
    r.num("crown.base_of_height")
    r.range("trunk.height_m")
    r.range("trunk.height_m")
    assert r.read == frozenset({"crown.base_of_height", "trunk.height_m"})


def test_reader_missing_key_names_it():
    r = reader()
    with pytest.raises(ManifestError, match=r"params: no 'crown\.nope'") as exc:
        r.num("crown.nope")
    assert str(TINY) in str(exc.value)
    assert "crown.nope" not in r.read


def test_reader_wrong_kind_raises():
    r = reader()
    with pytest.raises(ManifestError, match="trunk.height_m"):
        r.num("trunk.height_m")
    with pytest.raises(ManifestError, match="crown.base_of_height"):
        r.range("crown.base_of_height")
    with pytest.raises(ManifestError, match="leaf.kinds"):
        r.choices("leaf.kinds")
    with pytest.raises(ManifestError, match="crown.base_of_height"):
        r.names("crown.base_of_height")
    with pytest.raises(ManifestError, match="trunk.height_m"):
        r.points("trunk.height_m")


def test_int_rejects_fraction(tmp_path):
    r = reader()
    with pytest.raises(ManifestError, match="crown.base_of_height"):
        r.int("crown.base_of_height")
    with pytest.raises(ManifestError, match="trunk.height_m"):
        r.int_range("trunk.height_m")
    data = tiny_data()
    data["params"]["branch.count_choices"]["value"] = [3, 4.5]
    with pytest.raises(ManifestError, match="branch.count_choices"):
        RuleReader(load_rules(write(tmp_path, data))).int_choices("branch.count_choices")


def test_int_returns_python_ints():
    r = reader()
    assert type(r.int("branch.tiers")) is int
    assert all(type(v) is int for v in r.int_choices("branch.count_choices"))
    assert all(type(v) is int for v in r.int_range("branch.angle_deg"))


def test_range_must_not_run_backwards(tmp_path):
    data = tiny_data()
    data["params"]["trunk.height_m"]["value"] = [2.4, 1.6]
    with pytest.raises(ManifestError, match="trunk.height_m"):
        RuleReader(load_rules(write(tmp_path, data))).range("trunk.height_m")


def test_metres_converts():
    r = reader()
    assert r.metres("trunk.girth_m", 0.5) == 0.11
    assert r.metres_range("trunk.height_m", 0.5) == (0.8, 1.2)
    assert r.read == frozenset({"trunk.girth_m", "trunk.height_m"})


# --- kit-wide rules, tags and ranges


def buds_data() -> dict:
    return json.loads(TINY_BUDS.read_text())


def load_buds(tmp_path: Path, data: dict) -> GrowthRules:
    return load_rules(write(tmp_path, data), kit=KIT_RULES)


def add_fitted(data: dict, n: int) -> None:
    """Add `n` more fitted params, each cited to an existing row and fitted to one range."""
    for i in range(n):
        data["params"][f"extra.fit{i}"] = {
            "value": 0.1,
            "rows": ["apical.control"],
            "tag": "fitted",
            "fit": {"ranges": ["spread_of_height"], "seeds": [1, 2, 3, 4]},
        }


def test_kit_file_loads_alone():
    with pytest.raises(ManifestError, match="species"):
        load_rules(KIT_RULES)
    rules = load_rules(TINY_BUDS, kit=KIT_RULES)
    assert "leonardo.exponent" in rules.kit_keys
    assert "leonardo.exponent" in rules.params
    assert "apical.control" not in rules.kit_keys
    assert RuleReader(rules).num("leonardo.exponent") == 2.0


def test_kit_params_are_tagged_model_or_fact():
    rules = load_rules(TINY_BUDS, kit=KIT_RULES)
    tags = {key: rules.params[key].tag for key in rules.kit_keys}
    assert set(tags.values()) == {"model", "fact"}
    assert tags["leonardo.exponent"] == "fact"
    assert tags["light.shadow_a"] == "model"


def test_species_cannot_set_a_kit_key(tmp_path):
    data = buds_data()
    data["params"]["leonardo.exponent"] = {
        "value": 2.5, "rows": ["internode.length"], "tag": "measured"}
    with pytest.raises(ManifestError, match=r"leonardo\.exponent.*kit"):
        load_buds(tmp_path, data)


def test_species_cannot_reuse_a_kit_source_or_row(tmp_path):
    rules = load_rules(TINY_BUDS, kit=KIT_RULES)
    species_rows = {row["id"] for row in buds_data()["rows"]}
    kit_source = next(k for k in rules.sources if k not in {"A", "B"})
    kit_row = next(r for r in rules.rows if r not in species_rows)
    data = buds_data()
    data["sources"].append({"key": kit_source, "url": "https://example.org/x", "gives": "x"})
    with pytest.raises(ManifestError, match=rf"{kit_source}.*kit"):
        load_buds(tmp_path, data)
    data = buds_data()
    data["rows"][0].update(id=kit_row, level=kit_row.split(".")[0])
    data["params"]["leaves.divergence_deg"]["rows"] = [kit_row]
    with pytest.raises(ManifestError, match=rf"{kit_row}.*kit"):
        load_buds(tmp_path, data)


def test_every_param_needs_a_tag_with_the_kit(tmp_path):
    data = buds_data()
    del data["params"]["internode.length_m"]["tag"]
    with pytest.raises(ManifestError, match=r"params\.internode\.length_m.*tag"):
        load_buds(tmp_path, data)


def test_unknown_tag_raises(tmp_path):
    data = buds_data()
    data["params"]["internode.length_m"]["tag"] = "guess"
    with pytest.raises(ManifestError, match=r"params\.internode\.length_m\.tag.*'guess'"):
        load_buds(tmp_path, data)


def test_fitted_params_capped_by_ranges(tmp_path):
    data = buds_data()  # 1 fitted param, 2 ranges
    assert load_buds(tmp_path, data).params["apical.control"].fit is not None
    add_fitted(data, 1)  # 2 fitted, 2 ranges: at the cap
    load_buds(tmp_path, data)
    add_fitted(data, 2)  # 3 fitted (names repeat, so 3 distinct), 2 ranges
    with pytest.raises(ManifestError, match="fitted"):
        load_buds(tmp_path, data)


def test_fit_names_existing_ranges_and_seeds_1_to_4(tmp_path):
    data = buds_data()
    data["params"]["apical.control"]["fit"]["ranges"] = ["spread_of_height", "lai"]
    with pytest.raises(ManifestError, match=r"fit\.ranges\[1\].*'lai'"):
        load_buds(tmp_path, data)
    data = buds_data()
    data["params"]["apical.control"]["fit"]["seeds"] = [1, 2]
    with pytest.raises(ManifestError, match=r"fit\.seeds.*1, 2, 3, 4"):
        load_buds(tmp_path, data)
    data = buds_data()
    del data["params"]["apical.control"]["fit"]
    with pytest.raises(ManifestError, match=r"params\.apical\.control.*fit"):
        load_buds(tmp_path, data)
    data = buds_data()
    data["params"]["internode.length_m"]["fit"] = {
        "ranges": ["spread_of_height"], "seeds": [1, 2, 3, 4]}
    with pytest.raises(ManifestError, match=r"params\.internode\.length_m.*fit"):
        load_buds(tmp_path, data)
    fit = load_buds(tmp_path, buds_data()).params["apical.control"].fit
    assert fit.ranges == ("spread_of_height", "crown_base_of_height")
    assert fit.seeds == (1, 2, 3, 4)


def test_ranges_are_cited(tmp_path):
    rules = load_buds(tmp_path, buds_data())
    rng = rules.ranges["spread_of_height"]
    assert (rng.key, rng.lo, rng.hi, rng.rows) == ("spread_of_height", 0.5, 0.9, ("size.spread",))
    data = buds_data()
    data["ranges"]["spread_of_height"]["rows"] = ["size.nope"]
    with pytest.raises(ManifestError, match=r"ranges\.spread_of_height\.rows\[0\].*size\.nope"):
        load_buds(tmp_path, data)
    data = buds_data()
    data["ranges"]["spread_of_height"]["rows"] = []
    with pytest.raises(ManifestError, match=r"ranges\.spread_of_height\.rows"):
        load_buds(tmp_path, data)
    data = buds_data()
    data["ranges"]["spread_of_height"].update(lo=0.9, hi=0.5)
    with pytest.raises(ManifestError, match=r"ranges\.spread_of_height.*0\.9.*0\.5"):
        load_buds(tmp_path, data)
    data = buds_data()
    data["ranges"]["lai"] = {"lo": 1.0, "hi": 2.0, "rows": ["size.spread"]}
    with pytest.raises(ManifestError, match=r"ranges\.lai.*spread_of_height"):
        load_buds(tmp_path, data)


def test_a_row_cited_only_by_a_range_is_used(tmp_path):
    data = buds_data()
    data["ranges"].pop("crown_base_of_height")
    data["params"]["apical.control"]["fit"]["ranges"] = ["spread_of_height"]
    with pytest.raises(ManifestError, match=r"size\.crown-base.*no param cites it"):
        load_buds(tmp_path, data)


@pytest.mark.parametrize("turn", [0, -10, 180.5, 200])
def test_divergence_must_be_a_turn(tmp_path, turn):
    data = buds_data()
    data["params"]["leaves.divergence_deg"]["value"] = turn
    with pytest.raises(ManifestError, match=r"leaves\.divergence_deg.*\(0, 180\]"):
        load_buds(tmp_path, data)


@pytest.mark.parametrize("turn", [180, 137.5, 90, 0.5])
def test_divergence_inside_the_turn_loads(tmp_path, turn):
    data = buds_data()
    data["params"]["leaves.divergence_deg"]["value"] = turn
    assert RuleReader(load_buds(tmp_path, data)).num("leaves.divergence_deg") == turn


def test_tag_vocabulary(tmp_path):
    rules = load_rules(TINY_BUDS, kit=KIT_RULES)
    for key, param in rules.params.items():
        allowed = {"model", "fact"} if key in rules.kit_keys else {"measured", "fitted", "budget", "model"}
        assert param.tag in allowed, key
    data = buds_data()
    data["params"]["internode.length_m"]["tag"] = "fact"
    with pytest.raises(ManifestError, match=r"tag.*'fact'"):
        load_buds(tmp_path, data)


def test_sha256_is_the_file():
    rules = load_rules(TINY_BUDS, kit=KIT_RULES)
    assert rules.sha256 == hashlib.sha256(TINY_BUDS.read_bytes()).hexdigest()
    assert rules.kit_sha256 == hashlib.sha256(KIT_RULES.read_bytes()).hexdigest()


# --- a `fact` or `measured` tag needs a cite; a buds-v2 file has no tube sides


def uncite(row: dict) -> None:
    row["source"] = "none: a placeholder, not from field notes"


@pytest.mark.parametrize("species", ["tiny", "tiny-buds"])
def test_measured_row_without_cite_fails_the_load(tmp_path, species):  # guard
    """A `measured` param states where its number came from, so each row it cites carries a `[key]`
    cite. Planted on two different species files."""
    data = json.loads((TINY.parent / f"{species}.rules.json").read_text())
    key = next(k for k, p in data["params"].items() if p["tag"] == "measured")
    row = next(r for r in data["rows"] if r["id"] == data["params"][key]["rows"][0])
    uncite(row)
    with pytest.raises(ManifestError, match=rf"params\.{key}.*{row['id']}.*cite"):
        load_buds(tmp_path, data)


def test_fact_row_without_cite_fails_the_load(tmp_path):  # guard
    """The kit file's `fact` params follow the same rule: a copy of the kit file whose fact row
    cites nothing does not load."""
    kit = json.loads(KIT_RULES.read_text())
    row = next(r for r in kit["rows"] if r["id"] == "leonardo.exponent")
    uncite(row)
    path = tmp_path / "kit.rules.json"
    path.write_text(json.dumps(kit))
    with pytest.raises(ManifestError, match=r"leonardo\.exponent.*cite"):
        load_rules(TINY_BUDS, kit=path)


@pytest.mark.parametrize(("key", "tag"), [("roots.count", "model"), ("lite.leaf_thin", "budget")])
def test_model_and_budget_rows_may_cite_nothing(tmp_path, key, tag):
    """The rule is for the tags that claim a source: `budget` (a cost limit) and `model` (no
    source) rows say plainly that they have none."""
    data = buds_data()
    param = data["params"][key]
    assert param["tag"] == tag
    uncite(next(r for r in data["rows"] if r["id"] == param["rows"][0]))
    load_buds(tmp_path, data)


REPO = Path(__file__).resolve().parents[3]
# Every rules file the kit ships: the bars' own files and their scene copies, the kit file and
# the test fixtures. Found by search, so a new file is loaded without editing this list.
SHIPPED = sorted(p for d in ("bars", "src", "test") for p in (REPO / d).rglob("*.rules.json"))


def load_shipped(path: Path, *, kit: Path = KIT_RULES) -> GrowthRules:
    """Load a rules file as the kit would: the kit file as the kit, any other as a species."""
    return load_rules(TINY_BUDS, kit=path) if path == kit else load_rules(path, kit=kit)


def test_the_shipped_rules_files_are_all_found():
    names = {p.relative_to(REPO).as_posix() for p in SHIPPED}
    assert KIT_RULES in SHIPPED
    assert {"bars/02-redbud/redbud.rules.json", "bars/02-redbud/scene/rules/redbud.rules.json",
            "bars/04-sweetgum/sweetgum.rules.json",
            "bars/04-sweetgum/scene/rules/sweetgum.rules.json"} <= names


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_shipped_rules_file_loads(path):  # guard
    """Each shipped rules file loads under the current rules, so a rule added to the loader
    (such as the cite rule) is met by every file, not only the ones a grower test reads."""
    load_shipped(path)


@pytest.mark.parametrize("name", ["02-redbud/redbud", "04-sweetgum/sweetgum"])
def test_a_shipped_file_with_an_uncited_measured_row_does_not_load(tmp_path, name):
    """The plant for the guard above: a copy of a real bar file whose `measured` row cites
    nothing fails `load_shipped`, so a shipped file in that state fails the guard."""
    data = json.loads((REPO / "bars" / f"{name}.rules.json").read_text())
    key = next(k for k, p in data["params"].items() if p["tag"] == "measured")
    row = next(r for r in data["rows"] if r["id"] == data["params"][key]["rows"][0])
    uncite(row)
    with pytest.raises(ManifestError, match=rf"params\.{key}.*{row['id']}.*cite"):
        load_shipped(write(tmp_path, data))


def test_kit_leaf_season_is_a_model_not_a_fact():
    """No source was found for `leaves.on_current_year` so it is a model constant."""
    rules = load_rules(TINY_BUDS, kit=KIT_RULES)
    assert rules.params["leaves.on_current_year"].tag == "model"


def test_a_species_file_with_lite_tube_sides_but_no_tube_sides_fails_the_load(tmp_path):
    """No `tubes.sides` makes a buds-v2 file, whose sides come from the view. A stranded
    `lite.tube_sides` param would be a v1 key nobody reads."""
    data = buds_data()
    del data["params"]["tubes.sides"]
    with pytest.raises(ManifestError, match=r"lite\.tube_sides.*buds-v2"):
        load_buds(tmp_path, data)


@pytest.mark.parametrize("row", ["tubes.sides", "lite.tree-line"])
def test_a_tube_sides_row_in_a_buds_v2_file_fails_the_load(tmp_path, row):
    """A buds-v2 file (no `tubes.sides` param) may not keep the v1 sides rows, even as notes."""
    data = buds_data()
    for key in ("tubes.sides", "lite.tube_sides"):
        del data["params"][key]
    data["rows"] = [r for r in data["rows"] if r["id"] not in TUBE_SIDES_ROWS or r["id"] == row]
    data["notes"] = [*data["notes"], row]
    with pytest.raises(ManifestError, match=rf"{row}.*buds-v2"):
        load_buds(tmp_path, data)


def test_a_buds_v2_file_without_the_sides_rows_loads(tmp_path):
    data = buds_data()
    for key in ("tubes.sides", "lite.tube_sides"):
        del data["params"][key]
    data["rows"] = [r for r in data["rows"] if r["id"] not in ("tubes.sides", "lite.tree-line")]
    load_buds(tmp_path, data)


def test_a_buds_v1_file_with_both_sides_params_still_loads(tmp_path):
    load_buds(tmp_path, buds_data())
