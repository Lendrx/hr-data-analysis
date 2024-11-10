import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Beispielwerte für die Felder
job_titles = ["Data Scientist", "Software Engineer", "Product Manager", "HR Specialist", "Sales Executive"]
companies = ["Company A", "Company B", "Company C"]
departments = ["IT", "Marketing", "HR", "Sales", "Finance"]
names = ["Lena Müller", "Maximilian Weber", "Laura Fischer", "Jonas Schäfer", "Mia Hoffmann",
         "Alexander Wagner", "Sophie Keller", "Lukas Neumann", "Hannah Meier", "Felix Braun",
         "Lea Richter", "Paul Zimmermann", "Nina Krause", "Leon Becker", "Emilia Klein",
         "Marie Wolf", "Jan Schröder", "Sara Frank", "Daniel Weiß", "Lara Schmitt"]

# Funktion zum zufälligen Generieren von Daten für Mitarbeiter
def generate_employee_data(num_records=100):
    data = []
    for i in range(num_records):
        employee_id = 1000 + i  # Personalnummer
        birth_date = datetime.now() - timedelta(days=random.randint(9000, 20000))  # Geburtsdatum zwischen 25 und 55 Jahren alt
        entry_date = datetime.now() - timedelta(days=random.randint(100, 5000))  # Eintrittsdatum in den letzten 15 Jahren
        # 20% der Mitarbeiter haben ein Austrittsdatum
        exit_date = entry_date + timedelta(days=random.randint(1, 2000)) if random.random() < 0.2 else None
        # 10% der Mitarbeiter haben ein Wiedereintrittsdatum
        reentry_date = exit_date + timedelta(days=random.randint(30, 1000)) if exit_date and random.random() < 0.1 else None
        job_title = random.choice(job_titles)
        company = random.choice(companies)
        name = random.choice(names)  # Zufälliger Name aus der Liste
        
        data.append({
            "Name": name,
            "Personalnummer": employee_id,
            "Geburtsdatum": birth_date.strftime("%Y-%m-%d"),
            "Eintrittsdatum": entry_date.strftime("%Y-%m-%d"),
            "Austrittsdatum": exit_date.strftime("%Y-%m-%d") if exit_date else None,
            "Wiedereintritt": reentry_date.strftime("%Y-cd raw%m-%d") if reentry_date else None,
            "Berufsbezeichnung": job_title,
            "Gesellschaft": company
        })
    return pd.DataFrame(data)

# Generiere die Daten
employee_data = generate_employee_data(100)

# Speichere die Daten in einer CSV-Datei
csv_path = "/Users/lennartdreisbach/Data Science/Github/personal_report_ml/data/raw/mitarbeiterdaten.csv"
employee_data.to_csv(csv_path, index=False)

print(f"Die CSV-Datei wurde erfolgreich unter {csv_path} gespeichert.")
