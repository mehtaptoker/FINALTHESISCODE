#####This code is made with help of G00GLE GEMINI############
import json
import os

def read_constraints_file(file_path):
    """
    Leest een JSON constraints bestand en print de inhoud.

    Args:
        file_path (str): Het pad naar het JSON bestand.
    """
    print(f"--- Proberen bestand te lezen: {file_path} ---")

    # Controleer of het bestand bestaat
    if not os.path.exists(file_path):
        print(f"FOUT: Bestand niet gevonden op het opgegeven pad.")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        print("\n--- Bestandsinhoud succesvol gelezen ---")
        print(json.dumps(data, indent=2)) # Print de data mooi geformatteerd

        # Specifiek controleren op 'boundary_margin'
        if isinstance(data, dict):
            if "boundary_margin" in data:
                margin = data["boundary_margin"]
                print(f"\n---> 'boundary_margin' gevonden: {margin} (Type: {type(margin)})")
            else:
                print("\n---> SLEUTEL 'boundary_margin' NIET GEVONDEN in dit bestand.")
        else:
            print("\n---> JSON bevat geen dictionary op het hoogste niveau.")

    except json.JSONDecodeError as e:
        print(f"\nFOUT: Kon de JSON data niet parsen.")
        print(f"   Controleer of het bestand geldige JSON bevat.")
        print(f"   Foutmelding: {e}")
    except Exception as e:
        print(f"\nFOUT: Er is een onverwachte fout opgetreden: {e}")

if __name__ == "__main__":
    # Definieer het pad naar je constraints bestand
    # Pas dit pad aan als het bestand ergens anders staat
    constraints_file = r"C:\Users\Mmdocks\Desktop\CODE\GearboxRL\GearRL\data\Example1_constraints.json"

    # Roep de functie aan
    read_constraints_file(constraints_file)

