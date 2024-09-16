import sqlite3
import os
import json
from dataclasses import dataclass
from typing import Optional

from .constants import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, 'spellcaster.db')
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class Spell:
    name: str
    id: Optional[int] = None
    action_id: Optional[int] = None


@dataclass
class Action:
    id: Optional[int]
    name: str
    function: str
    payload: Optional[str]


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_tables():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spells (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                action_id INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                function TEXT NOT NULL,
                payload TEXT
            )
        ''')
        conn.commit()

create_tables()


def get_all_spells() -> list[Spell]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM spells')
        spells = cursor.fetchall()
        return [Spell(id=row[0], name=row[1], action_id=row[2]) for row in spells]


def add_spell(spell: Spell):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO spells (name, action_id) VALUES (?, ?)', (spell.name, spell.action_id))
        conn.commit()


def get_spell(spell_name: str) -> Optional[Spell]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM spells WHERE name = ?', (spell_name,))
        row = cursor.fetchone()
        if row:
            return Spell(id=row[0], name=row[1], action_id=row[2])
        return None
    

def get_spell_by_id(spell_id: int) -> Optional[Spell]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM spells WHERE id = ?', (spell_id,))
        row = cursor.fetchone()
        if row:
            return Spell(id=row[0], name=row[1], action_id=row[2])
        return None
    

def update_spell(spell: Spell):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE spells SET name = ?, action_id = ? WHERE id = ?', (spell.name, spell.action_id, spell.id))
        conn.commit()


def delete_spell(spell: Spell):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM spells WHERE id = ?', (spell.id,))
        conn.commit()


def add_action(action: Action):
    with get_connection() as conn:
        cursor = conn.cursor()
        payload_json = json.dumps(action.payload) if action.payload is not None else None
        cursor.execute('INSERT INTO actions (name, function, payload) VALUES (?, ?, json(?))', (action.name, action.function, payload_json))
        conn.commit()


def get_all_actions() -> list[Action]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, function, json_extract(payload, "$") as payload FROM actions')
        actions = cursor.fetchall()
        return [Action(id=row[0], name=row[1], function=row[2], payload=row[3]) for row in actions]


def get_action(action_id: int) -> Optional[Action]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, function, json_extract(payload, "$") as payload FROM actions WHERE id = ?', (action_id,))
        row = cursor.fetchone()
        if row:
            return Action(id=row[0], name=row[1], function=row[2], payload=row[3])
        return None


def update_action(action: Action):
    with get_connection() as conn:
        cursor = conn.cursor()
        payload_json = json.dumps(action.payload) if action.payload is not None else None
        cursor.execute('UPDATE actions SET name = ?, function = ?, payload = json(?) WHERE id = ?', (action.name, action.function, payload_json, action.id))
        conn.commit()


def delete_action(action: Action):
    with get_connection() as conn:
        cursor = conn.cursor()
        # Check if any spells reference this action
        cursor.execute('SELECT COUNT(*) FROM spells WHERE action_id = ?', (action.id,))
        count = cursor.fetchone()[0]
        if count > 0:
            raise ValueError(f"Cannot delete action {action.id} because it is referenced by {count} spell(s).")
        cursor.execute('DELETE FROM actions WHERE id = ?', (action.id,))
        conn.commit()


def manage_db(action: str, table: str, entry: Optional[dict] = None, entry_id: Optional[int] = None):
    if action == 'view':
        if table == 'spells':
            spells = get_all_spells()
            for spell in spells:
                print(spell)
        elif table == 'actions':
            actions = get_all_actions()
            for action in actions:
                print(action)
        else:
            print(f"Unknown table: {table}")
    elif action == 'add' and entry:
        if table == 'spells':
            spell = Spell(id=None, name=entry['name'], action_id=entry.get('action_id'))
            add_spell(spell)
        elif table == 'actions':
            action = Action(id=None, name=entry['name'], function=entry['function'], payload=entry.get('payload'))
            add_action(action)
        else:
            print(f"Unknown table: {table}")
    elif action == 'update' and entry:
        if table == 'spells':
            spell = Spell(id=entry['id'], name=entry['name'], action_id=entry.get('action_id'))
            update_spell(spell)
        elif table == 'actions':
            action = Action(id=entry['id'], name=entry['name'], function=entry['function'], payload=entry.get('payload'))
            update_action(action)
        else:
            print(f"Unknown table: {table}")
    elif action == 'delete' and entry_id is not None:
        if table == 'spells':
            spell = get_spell_by_id(entry_id)
            if spell:
                delete_spell(spell)
            else:
                print(f"No spell found with id: {entry_id}")
        elif table == 'actions':
            action = get_action(entry_id)
            if action:
                try:
                    delete_action(action)
                except ValueError as e:
                    print(e)
            else:
                print(f"No action found with id: {entry_id}")
        else:
            print(f"Unknown table: {table}")
    else:
        print(f"Invalid action: {action} or missing parameters")
