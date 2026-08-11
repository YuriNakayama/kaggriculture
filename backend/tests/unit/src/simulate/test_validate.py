"""ActionValidator catches what the engine silently discards."""

from __future__ import annotations

from typing import Any

from simulate.validate import ActionValidator

BOARD = 10


def _obs(
    *,
    step: int = 0,
    farmer: tuple[int, int] = (2, 2),
    hands: list[tuple[int, int]] | None = None,
    seeds: dict[str, int] | None = None,
    unlocked: list[str] | None = None,
) -> dict[str, Any]:
    """A minimal observation with only the fields the validator reads."""
    return {
        "player": 0,
        "step": step,
        "farms": [
            {
                "money": 3000,
                "tiles": [[None] * BOARD for _ in range(BOARD)],
                "farmer": list(farmer),
                "hands": [list(h) for h in (hands or [])],
                "unlocked_quadrants": unlocked if unlocked is not None else ["NW"],
            },
            {},
        ],
        "private": {"shed": {}, "seeds": seeds or {}, "inventories": [{}]},
    }


def _kinds(validator: ActionValidator) -> list[str]:
    return [issue.kind for issue in validator.issues]


def _validate(action: dict[str, Any], **obs_kwargs: Any) -> ActionValidator:
    validator = ActionValidator(player=0)
    validator.validate(_obs(**obs_kwargs), action)
    return validator


def test_clean_action_produces_no_issues() -> None:
    validator = _validate(
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 1]]}
    )

    assert validator.issues == ()


def test_eleventh_market_order_is_reported() -> None:
    # The engine truncates with q[:max_orders] — no warning, no error.
    orders = [["SELL", "WHEAT", 1] for _ in range(11)]

    validator = _validate({"farmer": ["PASS"], "hands": [], "market": orders})

    assert "market_orders_truncated" in _kinds(validator)
    assert "11 orders exceeds" in str(validator.issues[0])


def test_ten_market_orders_are_allowed() -> None:
    orders = [["SELL", "WHEAT", 1] for _ in range(10)]

    validator = _validate({"farmer": ["PASS"], "hands": [], "market": orders})

    assert validator.issues == ()


def test_unknown_unit_op_is_reported() -> None:
    validator = _validate({"farmer": ["HARVSET"], "hands": [], "market": []})

    assert "unknown_unit_op" in _kinds(validator)


def test_shed_op_away_from_access_tile_is_reported() -> None:
    validator = _validate(
        {"farmer": ["PICKUP", "WHEAT"], "hands": [], "market": []}, farmer=(0, 0)
    )

    assert "shed_not_adjacent" in _kinds(validator)


def test_shed_op_on_access_tile_is_accepted() -> None:
    # (4,4) / (5,4) / (4,5) / (5,5) on a 10x10 board.
    validator = _validate(
        {"farmer": ["PICKUP", "WHEAT"], "hands": [], "market": []}, farmer=(4, 4)
    )

    assert validator.issues == ()


def test_plant_beyond_seed_count_is_reported() -> None:
    # Two units both PLANT WHEAT with one seed: the engine drops BOTH.
    validator = _validate(
        {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]], "market": []},
        hands=[(3, 3)],
        seeds={"WHEAT": 1},
    )

    kinds = _kinds(validator)
    assert "plant_over_seeds" in kinds
    assert "drops all of them" in str(validator.issues[kinds.index("plant_over_seeds")])


def test_plant_within_seed_count_is_accepted() -> None:
    validator = _validate(
        {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]], "market": []},
        hands=[(3, 3)],
        seeds={"WHEAT": 2},
    )

    assert validator.issues == ()


def test_tile_op_in_locked_quadrant_is_reported() -> None:
    # (7,2) is NE, which starts locked.
    validator = _validate(
        {"farmer": ["WATER"], "hands": [], "market": []}, farmer=(7, 2)
    )

    assert "locked_quadrant" in _kinds(validator)


def test_move_off_board_is_reported() -> None:
    validator = _validate(
        {"farmer": ["NORTH"], "hands": [], "market": []}, farmer=(0, 0)
    )

    assert "move_off_board" in _kinds(validator)


def test_hand_count_mismatch_is_reported() -> None:
    validator = _validate(
        {"farmer": ["PASS"], "hands": [], "market": []}, hands=[(3, 3), (4, 3)]
    )

    assert "hand_count_mismatch" in _kinds(validator)


def test_market_order_missing_quantity_is_reported() -> None:
    # _parse_order returns None for len(order) < 3.
    validator = _validate(
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT"]]}
    )

    assert "order_missing_quantity" in _kinds(validator)


def test_buy_product_rejects_unbuyable_item() -> None:
    validator = _validate(
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "MILK", 1]]}
    )

    kinds = _kinds(validator)
    assert "order_bad_item" in kinds
    assert "only WHEAT and FERTILIZER" in str(validator.issues[0])


def test_buy_product_accepts_wheat_and_fertilizer() -> None:
    validator = _validate(
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_PRODUCT", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 1]],
        }
    )

    assert validator.issues == ()


def test_zero_quantity_order_is_reported() -> None:
    validator = _validate(
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 0]]}
    )

    assert "order_bad_quantity" in _kinds(validator)


def test_non_dict_action_is_reported() -> None:
    validator = ActionValidator(player=0)
    validator.validate(_obs(), ["PASS"])

    assert "action_not_dict" in _kinds(validator)


def test_hire_and_buy_land_need_no_arguments() -> None:
    validator = _validate(
        {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["BUY_LAND"]]}
    )

    assert validator.issues == ()


def test_issue_count_is_bounded() -> None:
    validator = ActionValidator(player=0, max_issues=5)
    for step in range(20):
        validator.validate(_obs(step=step), {"farmer": ["NOPE"], "hands": []})

    assert len(validator.issues) == 5


def test_wrap_passes_the_action_through_unchanged() -> None:
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    validator = ActionValidator(player=0)

    wrapped = validator.wrap(lambda obs: action)

    assert wrapped(_obs()) is action
    assert validator.issues == ()


def test_wrap_records_issues_from_the_wrapped_agent() -> None:
    validator = ActionValidator(player=0)
    wrapped = validator.wrap(lambda obs: {"farmer": ["BOGUS"], "hands": []})

    wrapped(_obs())

    assert "unknown_unit_op" in _kinds(validator)


def test_placing_an_animal_on_a_pasture_is_legal_away_from_the_shed() -> None:
    """The engine resolves animal placement before the shed-drop path.

    `_apply_unit_action` matches `PLACE <animal>` against the tile's structure
    and returns, so a cow goes onto a pasture anywhere on the board. Treating
    every PLACE as a shed op reported this as discarded when it is not, and
    that false positive fired on four agent cases.
    """
    obs = _obs(farmer=(1, 1))
    obs["farms"][0]["tiles"][1][1] = {"kind": "PASTURE"}
    validator = ActionValidator(player=0)

    validator.validate(obs, {"farmer": ["PLACE", "COW"], "hands": [], "market": []})

    assert _kinds(validator) == []


def test_placing_an_item_away_from_the_shed_is_still_reported() -> None:
    """Non-animal PLACE falls through to the shed drop, which needs the tile."""
    obs = _obs(farmer=(1, 1))
    validator = ActionValidator(player=0)

    action = {"farmer": ["PLACE", "WHEAT", 3], "hands": [], "market": []}
    validator.validate(obs, action)

    assert "shed_not_adjacent" in _kinds(validator)


def test_placing_an_animal_on_an_occupied_pasture_is_reported() -> None:
    """An occupied structure fails the engine's match, so PLACE falls through."""
    obs = _obs(farmer=(1, 1))
    obs["farms"][0]["tiles"][1][1] = {"kind": "PASTURE", "animal": "COW"}
    validator = ActionValidator(player=0)

    validator.validate(obs, {"farmer": ["PLACE", "SHEEP"], "hands": [], "market": []})

    assert "shed_not_adjacent" in _kinds(validator)
