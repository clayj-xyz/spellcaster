from dotenv import dotenv_values
from requests import post

from spellcaster.actions_registry import register


@register("log something")
def log_something(something: str):
    print(something)


@register("turn on office light")
def turn_on_light():
    api_key = dotenv_values(".env")["HOMEASSISTANT_API_KEY"]
    url = "http://homeassistant.local:8123/api/services/light/turn_on"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"entity_id": "light.office_fan"}

    response = post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("Light turned on")
    else:
        raise RuntimeError(f"Failed to turn on light: {response.text}")


if __name__ == "__main__":
    turn_on_light()