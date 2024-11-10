import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import os
import json
from pathlib import Path
warnings.filterwarnings('ignore')

class HRAnalyzer:
    def __init__(self, input_file, output_dir='analysis_results'):
        """
        Initialisiert den HR Analyzer
        """
        self.df = pd.read_csv(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def save_plot(self, fig, filename):
        """
        Speichert eine Matplotlib-Figur
        """
        fig.savefig(self.output_dir / filename)
        plt.close(fig)

    def analyze_hr_data(self):
        """
        Führt die HR-Datenanalyse durch und speichert die Ergebnisse
        """
        # Datums-Spalten in datetime umwandeln
        date_columns = ['Geburtsdatum', 'Eintrittsdatum', 'Austrittsdatum']
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col])
        
        # Alter und Beschäftigungsdauer berechnen
        self.df['Alter_bei_Eintritt'] = (self.df['Eintrittsdatum'] - self.df['Geburtsdatum']).dt.days / 365.25
        self.df['Beschaeftigungsdauer'] = np.where(
            self.df['Austrittsdatum'].isna(),
            (datetime.now() - self.df['Eintrittsdatum']).dt.days / 365.25,
            (self.df['Austrittsdatum'] - self.df['Eintrittsdatum']).dt.days / 365.25
        )
        
        # Basis-Statistiken speichern
        self.results['basic_stats'] = {
            'total_employees': len(self.df),
            'average_age_at_entry': self.df['Alter_bei_Eintritt'].mean(),
            'average_employment_duration': self.df['Beschaeftigungsdauer'].mean(),
            'job_distribution': self.df['Berufsbezeichnung'].value_counts().to_dict()
        }
        
        return self.df

    def create_visualizations(self):
        """
        Erstellt und speichert Visualisierungen
        """
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # 1. Altersverteilung
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=self.df, x='Alter_bei_Eintritt', bins=20, color='skyblue')
        plt.title('Altersverteilung bei Eintritt', pad=20)
        plt.xlabel('Alter')
        plt.ylabel('Anzahl')
        plt.tight_layout()
        self.save_plot(fig, 'altersverteilung.png')
        
        # 2. Beschäftigungsdauer
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=self.df, x='Berufsbezeichnung', y='Beschaeftigungsdauer', color='lightgreen')
        plt.xticks(rotation=45, ha='right')
        plt.title('Beschäftigungsdauer nach Position', pad=20)
        plt.tight_layout()
        self.save_plot(fig, 'beschaeftigungsdauer.png')
        
        # 3. Eintritte pro Jahr
        self.df['Eintrittsjahr'] = self.df['Eintrittsdatum'].dt.year
        yearly_entries = self.df['Eintrittsjahr'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_entries.plot(kind='bar', color='lightcoral')
        plt.title('Eintritte pro Jahr', pad=20)
        plt.xlabel('Jahr')
        plt.ylabel('Anzahl')
        plt.tight_layout()
        self.save_plot(fig, 'eintritte_pro_jahr.png')
        
        # Speichere Visualisierungs-Statistiken
        self.results['visualization_stats'] = {
            'years_analyzed': yearly_entries.index.tolist(),
            'entries_per_year': yearly_entries.to_dict()
        }

    def prepare_ml_data(self):
        """
        Bereitet die Daten für das ML-Modell vor
        """
        # Feature Engineering
        self.df['Monat_Eintritt'] = self.df['Eintrittsdatum'].dt.month
        self.df['Jahr_Eintritt'] = self.df['Eintrittsdatum'].dt.year
        self.df['Alter_Kategorie'] = pd.cut(self.df['Alter_bei_Eintritt'], 
                                     bins=[0, 25, 35, 45, 100],
                                     labels=['Jung', 'Mittel', 'Erfahren', 'Senior'])
        
        # Label Encoding
        le = LabelEncoder()
        self.df['Berufsbezeichnung_encoded'] = le.fit_transform(self.df['Berufsbezeichnung'])
        
        # Features auswählen
        features = ['Monat_Eintritt', 'Jahr_Eintritt', 'Alter_bei_Eintritt', 
                   'Berufsbezeichnung_encoded']
        
        # Target-Variable
        self.df['Hat_gekuendigt'] = self.df['Austrittsdatum'].notna().astype(int)
        
        return self.df[features], self.df['Hat_gekuendigt']

    def train_ml_model(self, X, y):
        """
        Trainiert das ML-Modell und speichert die Ergebnisse
        """
        # Daten aufteilen
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model trainieren
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Vorhersagen
        y_pred = rf_model.predict(X_test)
        
        # Ergebnisse speichern
        self.results['model_performance'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': ['Monat_Eintritt', 'Jahr_Eintritt', 'Alter_bei_Eintritt', 'Berufsbezeichnung'],
            'importance': rf_model.feature_importances_
        })
        
        # Feature Importance Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature', color='lightblue')
        plt.title('Feature Importance', pad=20)
        plt.tight_layout()
        self.save_plot(fig, 'feature_importance.png')
        
        self.results['feature_importance'] = feature_importance.to_dict('records')
        
        return rf_model

    def generate_report(self):
        """
        Generiert einen detaillierten Bericht
        """
        report_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # HTML-Report erstellen
        html_report = f"""
        <html>
        <head>
            <title>HR Analyse Bericht - {report_time}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .stats {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>HR Analyse Bericht</h1>
            <p>Erstellt am: {report_time}</p>
            
            <h2>Basis-Statistiken</h2>
            <div class="stats">
                <p>Gesamtzahl Mitarbeiter: {self.results['basic_stats']['total_employees']}</p>
                <p>Durchschnittsalter bei Eintritt: {self.results['basic_stats']['average_age_at_entry']:.2f} Jahre</p>
                <p>Durchschnittliche Beschäftigungsdauer: {self.results['basic_stats']['average_employment_duration']:.2f} Jahre</p>
            </div>
            
            <h2>Visualisierungen</h2>
            <img src="altersverteilung.png" alt="Altersverteilung">
            <img src="beschaeftigungsdauer.png" alt="Beschäftigungsdauer">
            <img src="eintritte_pro_jahr.png" alt="Eintritte pro Jahr">
            <img src="feature_importance.png" alt="Feature Importance">
            
            <h2>Model Performance</h2>
            <div class="stats">
                <p>Modell-Genauigkeit: {self.results['model_performance']['accuracy']:.2%}</p>
            </div>
        </body>
        </html>
        """
        
        # Berichte speichern
        with open(self.output_dir / 'report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        with open(self.output_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        
        print(f"\nBerichte wurden gespeichert in: {self.output_dir}")

def main():
    """
    Hauptfunktion zur Ausführung der Analyse
    """
    print("Starting HR Data Analysis...")
    
    # Analyzer initialisieren
    analyzer = HRAnalyzer('/Users/lennartdreisbach/Data Science/Github/hr-data-analysis/hr-data-analysis/data/raw/mitarbeiterdaten.csv')
    
    # Analyse durchführen
    analyzer.analyze_hr_data()
    analyzer.create_visualizations()
    
    # ML-Modell trainieren
    X, y = analyzer.prepare_ml_data()
    analyzer.train_ml_model(X, y)
    
    # Bericht generieren
    analyzer.generate_report()
    
    print("Analyse abgeschlossen!")
    print(f"Alle Ergebnisse wurden im Ordner '{analyzer.output_dir}' gespeichert.")

if __name__ == "__main__":
    main()