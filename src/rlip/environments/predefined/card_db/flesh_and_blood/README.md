# Flesh and Blood Card Database

This directory contains the packaged card metadata used by the RLIP
Talishar-inspired simulation environment.

## Files

- `cards.json`: card definitions used for deck construction and simulation rules.
- `heroes.json`: hero metadata (life, intellect, and class bindings).
- `import_from_talishar.py`: converter for importing full upstream card exports.

## Card Record Schema

Each card in `cards.json` includes:

- `id`: stable unique ID.
- `name`: card name.
- `pitch`: resource pitch value.
- `cost`: play cost.
- `power`: attack value (when applicable).
- `defense`: defense value (when applicable).
- `type_line`: game type line (for example, `Attack Action - Warrior`).
- `card_types`: normalized category tags.
- `class`: class identity.
- `talent`: optional talent identity.
- `rarity`: card rarity.
- `set`: set code.
- `keywords`: keyword list.
- `text`: rules text.
- `legality`: legality metadata.

## Notes

This package ships a playable starter card pool suitable for RL simulation and
benchmarking workflows. The schema is designed so additional official card
records can be appended without code changes.

To import a larger card export:

```bash
python import_from_talishar.py --source /path/to/upstream_cards.json --out cards.json
```
