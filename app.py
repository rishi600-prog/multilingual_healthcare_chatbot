import streamlit as st
import torch
from deep_translator import GoogleTranslator
import speech_recognition as sr
from gtts import gTTS
import io
import joblib
import difflib
import numpy as np
import re
from typing import List, Tuple, Dict
import base64
from pathlib import Path
from simplet5 import SimpleT5
import streamlit as st
import speech_recognition as sr
import pyaudio
import wave
import threading
import io
import time
from typing import Optional
import numpy as np
# Set page configuration
st.set_page_config(
    page_title="MediChat AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 2rem;
            background: linear-gradient(to bottom right, #f0f8ff, #ffffff);
        }
        
        /* Header styling */
        h1 {
            text-align: center;
            color: #2E7EFF;
            font-size: 3rem !important;
            margin-bottom: 2rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            border-radius: 15px;
            background: rgba(255,255,255,0.9);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Cards */
        .stCard {
            border-radius: 20px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            padding: 2rem;
            margin: 1.5rem 0;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid var(--border-color);
            transition: transform 0.3s ease;
        }
        
        .stCard:hover {
            transform: translateY(-5px);
        }
        
        /* Dark mode variables */
        [data-theme="dark"] {
            --background-color: #1E1E2E;
            --text-color: #FFFFFF;
            --border-color: #404040;
        }
        
        /* Light mode variables */
        [data-theme="light"] {
            --background-color: #FFFFFF;
            --text-color: #333333;
            --border-color: #E0E0E0;
        }
        
        /* Custom title */
        .custom-title {
            font-size: 2.8rem;
            font-weight: 700;
            text-align: center;
            color: var(--text-color);
            margin-bottom: 2.5rem;
            letter-spacing: 1px;
        }
        
        /* Info boxes */
        .info-box {
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            background: linear-gradient(135deg, rgba(0, 123, 255, 0.1), rgba(0, 123, 255, 0.05));
            border: 1px solid rgba(0, 123, 255, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 25px;
            padding: 0.8rem 2.5rem;
            background: linear-gradient(45deg, #2E7EFF, #1E90FF);
            color: white;
            border: none;
            transition: all 0.3s ease;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            background: linear-gradient(45deg, #1E90FF, #00BFFF);
        }
        
        /* Audio player */
        audio {
            width: 100%;
            border-radius: 25px;
            margin: 1.5rem 0;
            background: linear-gradient(45deg, #f0f0f0, #e0e0e0);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Selectbox styling */
        .stSelectbox {
            border-radius: 15px;
            border: 2px solid #2E7EFF;
            padding: 10px;
        }
        
        /* Radio buttons */
        .stRadio > div {
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* Text input */
        .stTextInput>div>div>input {
            border-radius: 15px;
            border: 2px solid #2E7EFF;
            padding: 10px 15px;
            font-size: 1.1rem;
        }
        
        /* File uploader */
        .stFileUploader {
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 15px;
            border: 2px dashed #2E7EFF;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(45deg, #2E7EFF, #1E90FF);
            border-radius: 10px;
        }
        
        /* Emoji decorations */
        .emoji-decoration {
            font-size: 1.5rem;
            margin: 0 5px;
        }
        
        /* Recording indicator */
        .recording-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
            background: rgba(255, 0, 0, 0.1);
            border-radius: 15px;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
    """, unsafe_allow_html=True)
class LiveAudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_frames = []
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()
        
    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.audio_frames = []
        
        # Open audio stream
        self.stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.start()
    
    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and return the recorded audio as bytes"""
        if self.recording:
            self.recording = False
            self.record_thread.join()
            self.stream.stop_stream()
            self.stream.close()
            
            # Convert recorded frames to WAV format
            audio_bytes = io.BytesIO()
            with wave.open(audio_bytes, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_frames))
            
            return audio_bytes.getvalue()
        return None
    
    def _record(self):
        """Internal method to record audio frames"""
        while self.recording:
            try:
                data = self.stream.read(self.chunk_size)
                self.audio_frames.append(data)
            except Exception as e:
                print(f"Error recording audio: {e}")
                break
    
    def __del__(self):
        """Cleanup PyAudio"""
        self.p.terminate()
class MedicalChatbot:
    def __init__(self):
        """Initialize the medical chatbot with models and required components."""
        try:
            # Initialize T5 model instead of BioBERT
            self.qa_model = SimpleT5()
            self.qa_model.load_model(
                "t5",
                r"C:\Users\srish\Downloads\file (13)-20250306T171459Z-001\file (13)\kaggle\working\outputs\simplet5-epoch-9-train-loss-0.4146-val-loss-0.4159",
                use_gpu=False)
            
            # Load symptom prediction model
            self.rf_model = joblib.load(r'C:\Users\srish\Downloads\multilingual_chatbot\qa_model\random_forest_model.pkl')
            #self.translator = GoogleTranslator(source='auto', target='en')
            
     
            self.label_encoder = joblib.load(r'C:\Users\srish\Downloads\multilingual_chatbot\qa_model\label_encoder.pkl')
            

            
            self.symptoms = [
                'abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure',
                'altered_sensorium', 'anxiety', 'back_pain', 'belly_pain', 'blackheads',
                'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool',
                'blurred_and_distorted_vision', 'breathlessness', 'brittle_nails', 'bruising',
                'burning_micturition', 'chest_pain', 'chills', 'cold_hands_and_feets',
                'coma', 'congestion', 'constipation', 'continuous_feel_of_urine',
                'continuous_sneezing', 'cough', 'cramps', 'dark_urine', 'dehydration',
                'depression', 'diarrhoea', 'dischromic_patches', 'distention_of_abdomen',
                'dizziness', 'drying_and_tingling_lips', 'enlarged_thyroid', 'excessive_hunger',
                'extra_marital_contacts', 'family_history', 'fast_heart_rate', 'fatigue',
                'fluid_overload', 'fluid_overload.1', 'foul_smell_of_urine', 'headache',
                'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption',
                'increased_appetite', 'indigestion', 'inflammatory_nails', 'internal_itching',
                'irregular_sugar_level', 'irritability', 'irritation_in_anus', 'itching',
                'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy',
                'loss_of_appetite', 'loss_of_balance', 'loss_of_smell', 'malaise',
                'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum',
                'muscle_pain', 'muscle_wasting', 'muscle_weakness', 'nausea', 'neck_pain',
                'nodal_skin_eruptions', 'obesity', 'pain_behind_the_eyes',
                'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking',
                'palpitations', 'passage_of_gases', 'patches_in_throat', 'phlegm',
                'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes',
                'pus_filled_pimples', 'receiving_blood_transfusion',
                'receiving_unsterile_injections', 'red_sore_around_nose',
                'red_spots_over_body', 'redness_of_eyes', 'restlessness', 'runny_nose',
                'rusty_sputum', 'scurring', 'shivering', 'silver_like_dusting',
                'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech',
                'small_dents_in_nails', 'spinning_movements', 'spotting_urination',
                'stiff_neck', 'stomach_bleeding', 'stomach_pain', 'sunken_eyes',
                'sweating', 'swelled_lymph_nodes', 'swelling_joints',
                'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremeties',
                'swollen_legs', 'throat_irritation', 'toxic_look_(typhos)',
                'ulcers_on_tongue', 'unsteadiness', 'visual_disturbances', 'vomiting',
                'watering_from_eyes', 'weakness_in_limbs', 'weakness_of_one_body_side',
                'weight_gain', 'weight_loss', 'yellow_crust_ooze', 'yellow_urine',
                'yellowing_of_eyes', 'yellowish_skin'
            ]
            self.symptoms_set = set(self.symptoms)
            
            # Initialize the symptom variations dictionary
            self.init_symptom_variations()
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise
    def init_symptom_variations(self):
        """Initialize the comprehensive symptom variations dictionary"""
        self.symptom_variations = {
        'abdominal_pain': ['stomach pain', 'tummy pain', 'belly pain', 'stomach ache', 'abdominal ache', 'gut pain', 'digestive pain', 'stomach cramps', 'abdominal cramps', 'belly ache', 'gastrointestinal pain', 'visceral pain'],
        
        'abnormal_menstruation': ['irregular periods', 'menstrual problems', 'irregular menstruation', 'heavy periods', 'painful periods', 'menstrual irregularity', 'period problems', 'abnormal bleeding', 'menstrual changes', 'irregular bleeding', 'dysmenorrhea', 'menorrhagia'],
        
        'acidity': ['acid reflux', 'heartburn', 'indigestion', 'stomach acid', 'gastric acid', 'acid indigestion', 'sour stomach', 'pyrosis', 'gastric reflux', 'stomach burning', 'GERD', 'gastroesophageal reflux'],
        
        'acute_liver_failure': ['liver failure', 'hepatic failure', 'liver dysfunction', 'liver problems', 'hepatic dysfunction', 'liver damage', 'liver disease', 'hepatic damage', 'liver distress', 'severe liver problems', 'fulminant hepatic failure', 'acute hepatic failure'],
        
        'altered_sensorium': ['confusion', 'disorientation', 'mental confusion', 'altered consciousness', 'altered mental state', 'mental status change', 'confused state', 'disoriented', 'consciousness changes', 'mental state changes', 'cognitive changes', 'mental fog'],
        
        'anxiety': ['anxious', 'worry', 'nervousness', 'panic', 'stress', 'restlessness', 'unease', 'anxious feelings', 'nervous tension', 'apprehension', 'anxiety disorder', 'nervous anxiety'],
        
        'back_pain': ['backache', 'spine pain', 'lower back pain', 'upper back pain', 'back ache', 'spinal pain', 'dorsal pain', 'lumbar pain', 'back problems', 'back discomfort', 'lumbago', 'vertebral pain'],
        
        'belly_pain': ['abdominal discomfort', 'stomach ache', 'tummy ache', 'gut pain', 'abdominal cramps', 'stomach pain', 'abdominal pain', 'digestive pain', 'belly ache', 'stomach cramps', 'visceral pain', 'gastrointestinal pain'],
        
        'blackheads': ['comedones', 'acne', 'blocked pores', 'skin spots', 'pimples', 'black spots', 'clogged pores', 'acne spots', 'skin blemishes', 'facial spots', 'open comedones', 'blackhead acne'],
        
        'bladder_discomfort': ['urinary discomfort', 'bladder pain', 'urinary pain', 'bladder pressure', 'urinary urgency', 'bladder problems', 'urination pain', 'bladder irritation', 'urinary issues', 'painful urination', 'cystitis', 'urinary tract discomfort'],
        
        'blister': ['skin blister', 'bubble on skin', 'fluid-filled sac', 'skin bubble', 'vesicle', 'skin lesion', 'water blister', 'skin sore', 'raised skin', 'fluid pocket', 'pustule', 'bullae'],
        
        'blood_in_sputum': ['hemoptysis', 'bloody cough', 'coughing blood', 'blood in phlegm', 'bloody mucus', 'bloody spit', 'blood-stained sputum', 'pink frothy sputum', 'red sputum', 'blood in saliva', 'bloody expectoration', 'pulmonary hemorrhage'],
        
        'bloody_stool': ['blood in stool', 'rectal bleeding', 'bloody feces', 'intestinal bleeding', 'bleeding from rectum', 'red stool', 'blood in bowel movement', 'bloody bowel movement', 'melena', 'hematochezia', 'gastrointestinal bleeding', 'rectal blood'],
        
        'blurred_and_distorted_vision': ['blurry vision', 'distorted sight', 'fuzzy vision', 'unclear vision', 'visual disturbance', 'vision problems', 'poor vision', 'impaired vision', 'hazy vision', 'visual impairment', 'diplopia', 'visual abnormalities'],
        
        'breathlessness': ['shortness of breath', 'difficulty breathing', 'cant breathe', "can't breathe", 'trouble breathing', 'dyspnea', 'respiratory distress', 'breathing difficulty', 'labored breathing', 'air hunger', 'respiratory problems', 'breathing problems'],
        
        'brittle_nails': ['weak nails', 'fragile nails', 'breaking nails', 'splitting nails', 'crumbling nails', 'nail problems', 'damaged nails', 'weak fingernails', 'nail breakage', 'nail splitting', 'onychorrhexis', 'nail brittleness'],
        
        'bruising': ['bruises', 'contusion', 'black and blue marks', 'skin bruising', 'easy bruising', 'purple spots', 'blood spots', 'ecchymosis', 'skin marks', 'bruise marks', 'hematoma', 'purpura'],
        
        'burning_micturition': ['burning urination', 'painful urination', 'burning when peeing', 'urinary burning', 'dysuria', 'burning pee', 'painful peeing', 'urination pain', 'burning sensation urinating', 'stinging urination', 'urethral burning', 'painful micturition'],
        
        'chest_pain': ['chest discomfort', 'chest tightness', 'chest pressure', 'thoracic pain', 'angina', 'chest ache', 'breast pain', 'cardiac pain', 'heart pain', 'chest burning', 'precordial pain', 'substernal pain'],
        
        'chills': ['shivers', 'shivering', 'feeling cold', 'rigors', 'cold sensation', 'shaking chills', 'cold sweats', 'temperature chills', 'fever chills', 'body chills', 'rigor', 'chilling sensation'],
        
        'cold_hands_and_feets': ['cold extremities', 'cold limbs', 'cold hands', 'cold feet', 'poor circulation', 'cold fingers', 'cold toes', 'circulation problems', 'cold hands and feet', 'peripheral coldness', 'acrocyanosis', 'cold peripheries'],
        
        'coma': ['unconscious', 'unresponsive', 'loss of consciousness', 'unconscious state', 'comatose', 'vegetative state', 'deep unconsciousness', 'unresponsive state', 'brain death', 'persistent unconsciousness', 'stupor', 'unconsciousness'],
        
        'congestion': ['stuffy nose', 'nasal congestion', 'blocked nose', 'sinus congestion', 'nose blockage', 'stuffed up', 'nasal blockage', 'blocked sinuses', 'runny nose', 'sinus problems', 'rhinitis', 'nasal obstruction'],
        
        'constipation': ['hard stools', 'difficult bowel movements', 'infrequent bowel movements', 'bowel problems', 'cant poop', "can't poop", 'difficulty pooping', 'irregular bowel movements', 'trapped stool', 'hardened stool', 'costiveness', 'obstipation'],
        
        'continuous_feel_of_urine': ['frequent urination', 'constant need to pee', 'urinary urgency', 'bladder pressure', 'constant urge to urinate', 'frequent peeing', 'overactive bladder', 'urinary frequency', 'constant bathroom need', 'persistent urge to urinate', 'pollakiuria', 'urinary tenesmus'],
        
        'continuous_sneezing': ['frequent sneezing', 'constant sneezing', 'sneezing fits', 'repeated sneezing', 'excessive sneezing', 'persistent sneezing', 'sneezing attacks', 'uncontrolled sneezing', 'recurrent sneezing', 'chronic sneezing', 'paroxysmal sneezing', 'sternutation'],
        
        'cough': ['coughing', 'chronic cough', 'persistent cough', 'dry cough', 'wet cough', 'hacking cough', 'chesty cough', 'night cough', 'productive cough', 'recurring cough', 'tussis', 'bronchial cough'],
        
        'cramps': ['muscle cramps', 'muscle spasms', 'painful cramps', 'cramping pain', 'muscle contractions', 'muscle tightness', 'spasmodic pain', 'muscular cramps', 'severe cramps', 'painful spasms', 'myospasm', 'muscle contracture'],
        
        'dark_urine': ['brown urine', 'cola colored urine', 'discolored urine', 'brown pee', 'dark colored urine', 'abnormal urine color', 'concentrated urine', 'tea colored urine', 'amber urine', 'darkened urine', 'melanuria', 'choluria'],
        
        'dehydration': ['fluid loss', 'lack of water', 'thirsty', 'water loss', 'dry mouth', 'decreased fluids', 'body fluid loss', 'fluid depletion', 'dehydrated', 'insufficient fluids', 'hypovolemia', 'fluid deficit'],
        
        'depression': ['feeling down', 'depressed mood', 'sadness', 'low mood', 'feeling blue', 'melancholy', 'depressive symptoms', 'emotional depression', 'mental depression', 'psychological depression', 'clinical depression', 'major depression'],
        
        'diarrhoea': ['loose stools', 'watery stools', 'frequent bowel movements', 'diarrhea', 'runny stools', 'loose bowels', 'watery bowel movements', 'frequent diarrhea', 'liquid stools', 'intestinal urgency', 'enteritis', 'loose motions'],
        
        'dischromic_patches': ['skin discoloration', 'patchy skin', 'uneven skin tone', 'skin patches', 'pigment changes', 'discolored patches', 'skin color changes', 'melanin changes', 'hyperpigmentation', 'hypopigmentation', 'dyschromia', 'skin pigmentation'],
        
        'distention_of_abdomen': ['bloated stomach', 'swollen belly', 'abdominal swelling', 'bloating', 'stomach distention', 'abdominal bloating', 'swollen abdomen', 'enlarged belly', 'stomach swelling', 'abdominal enlargement', 'tympanites', 'meteorism'],
        
        'dizziness': ['vertigo', 'lightheaded', 'light headed', 'feeling faint', 'spinning sensation', 'unsteady', 'giddiness', 'wooziness', 'dizzy spells', 'loss of balance', 'vestibular disorder', 'disequilibrium'],
        
        'drying_and_tingling_lips': ['chapped lips', 'lip tingling', 'dry lips', 'parched lips', 'lip numbness', 'lip sensation', 'lip dryness', 'lip discomfort', 'lip problems', 'lip burning', 'cheilitis', 'lip paresthesia'],
        
        'enlarged_thyroid': ['goiter', 'thyroid swelling', 'neck swelling', 'thyroid enlargement', 'swollen thyroid', 'thyroid mass', 'neck mass', 'thyroid growth', 'thyroid problems', 'thyroid nodule', 'thyromegaly', 'struma'],
        
        'excessive_hunger': ['increased appetite', 'always hungry', 'constant hunger', 'extreme hunger', 'insatiable appetite', 'overeating', 'increased food cravings', 'constant eating', 'ravenous appetite', 'excessive appetite', 'polyphagia', 'hyperphagia'],
        
        'extra_marital_contacts': ['multiple partners', 'unsafe sex', 'risky sexual behavior', 'sexual contacts', 'casual partners', 'unprotected sex', 'sexual risk', 'multiple sexual partners', 'high risk behavior', 'promiscuity', 'sexual promiscuity', 'unsafe sexual contact'],
        
        'family_history': ['genetic history', 'hereditary factors', 'family medical history', 'inherited conditions', 'family health history', 'genetic factors', 'family disease history', 'hereditary conditions', 'family risk factors', 'genetic predisposition', 'familial disease', 'hereditary disease'],
        
        'fast_heart_rate': ['rapid heartbeat', 'racing heart', 'heart racing', 'rapid pulse', 'palpitations', 'increased heart rate', 'rapid heart rate', 'tachycardia', 'heart palpitations', 'accelerated heart rate', 'cardiac tachycardia', 'sinus tachycardia'],
        
        'fatigue': ['tired', 'exhausted', 'low energy', 'tiredness', 'exhaustion', 'weakness', 'lethargy', 'worn out', 'drained', 'no energy', 'chronic fatigue', 'asthenia'],
        
        'fluid_overload': ['fluid retention', 'water retention', 'edema', 'swelling', 'fluid buildup', 'excess fluid', 'body fluid retention', 'fluid accumulation', 'water buildup', 'fluid excess', 'hypervolemia', 'volume overload'],
        
        'fluid_overload.1': ['body swelling', 'water accumulation', 'excess body fluid', 'generalized edema', 'fluid swelling', 'water excess', 'body fluid excess', 'fluid congestion', 'water logging', 'systemic edema', 'anasarca', 'volume excess'],
        
        'foul_smell_of_urine': ['smelly urine', 'strong urine odor', 'bad smelling urine', 'urine odor', 'foul urine', 'malodorous urine', 'offensive urine smell', 'abnormal urine smell', 'unpleasant urine odor', 'pungent urine', 'fetid urine', 'bromhidrosis'],
        
        'headache': ['head pain', 'head ache', 'migraine', 'tension headache', 'severe headache', 'throbbing head', 'head pressure', 'cranial pain', 'head discomfort', 'persistent headache', 'cephalgia', 'cephalea'],
        
        'high_fever': ['fever', 'high temperature', 'elevated temperature', 'febrile', 'pyrexia', 'hyperthermia', 'temperature elevation', 'raised temperature', 'severe fever', 'acute fever', 'febrile state', 'hyperpyrexia'],
        
        'hip_joint_pain': ['hip pain', 'hip ache', 'joint pain in hip', 'hip discomfort', 'hip joint ache', 'hip arthralgia', 'painful hip', 'hip joint discomfort', 'hip joint inflammation', 'hip soreness', 'coxalgia', 'hip arthritis'],
        
        'history_of_alcohol_consumption': ['alcoholism', 'alcohol use', 'drinking history', 'alcohol abuse', 'heavy drinking', 'alcohol dependence', 'alcohol addiction', 'chronic alcohol use', 'alcohol problems', 'alcohol habit', 'ethanol abuse', 'alcohol use disorder'],
        
        'increased_appetite': ['eating more', 'overeating', 'excessive hunger', 'heightened appetite', 'constant hunger', 'increased food intake', 'excessive eating', 'enhanced appetite', 'more hungry than usual', 'increased food cravings', 'hyperphagia', 'polyphagia'],
        
        'indigestion': ['dyspepsia', 'upset stomach', 'stomach upset', 'digestive problems', 'stomach problems', 'gastric discomfort', 'stomach discomfort', 'digestive issues', 'heartburn', 'acid indigestion', 'gastric upset', 'gastrointestinal distress'],
        
        'inflammatory_nails': ['nail inflammation', 'infected nails', 'nail infection', 'swollen nails', 'nail problems', 'inflamed nails', 'nail bed inflammation', 'nail disease', 'nail disorder', 'paronychia', 'onychitis', 'nail bed infection'],
        
        'internal_itching': ['inner itching', 'inside itching', 'deep itch', 'internal pruritus', 'subcutaneous itching', 'beneath skin itching', 'deep scratch sensation', 'internal itch sensation', 'inner skin itching', 'deep skin itch', 'visceral pruritus', 'internal pruritus'],
        
        'irregular_sugar_level': ['unstable blood sugar', 'blood sugar fluctuation', 'variable glucose', 'uncontrolled diabetes', 'glucose instability', 'blood sugar swings', 'erratic blood sugar', 'diabetes problems', 'glycemic instability', 'blood sugar variation', 'dysglycemia', 'glycemic variability'],
        
        'irritability': ['easily annoyed', 'agitation', 'quick temper', 'mood swings', 'irritable mood', 'short temper', 'emotional irritability', 'crankiness', 'grumpiness', 'emotional sensitivity', 'dysphoria', 'emotional lability'],
        
        'irritation_in_anus': ['anal irritation', 'rectal irritation', 'anal itching', 'rectal discomfort', 'anal discomfort', 'pruritus ani', 'anal burning', 'rectal burning', 'perianal irritation', 'anal soreness', 'anal pruritus', 'perianal pruritus'],
        
        'itching': ['pruritus', 'scratching', 'skin itching', 'itchy skin', 'itch', 'skin irritation', 'itchy sensation', 'need to scratch', 'itchiness', 'skin itch', 'pruritic sensation', 'cutaneous pruritus'],
        
        'joint_pain': ['arthralgia', 'painful joints', 'joint ache', 'joint inflammation', 'joint discomfort', 'aching joints', 'joint soreness', 'articular pain', 'joint stiffness', 'painful movement', 'arthritis pain', 'joint arthralgia'],
        
        'knee_pain': ['painful knee', 'knee ache', 'knee discomfort', 'knee joint pain', 'knee soreness', 'painful knee joint', 'knee joint ache', 'knee problems', 'knee arthralgia', 'knee joint discomfort', 'gonalgia', 'knee arthritis'],
        
        'lack_of_concentration': ['poor focus', 'difficulty concentrating', 'unable to focus', 'poor concentration', 'attention problems', 'distracted easily', 'loss of focus', 'difficulty focusing', 'poor attention', 'concentration problems', 'attention deficit', 'cognitive difficulty'],
        
        'lethargy': ['sluggishness', 'lack of energy', 'drowsiness', 'low energy', 'fatigue', 'tiredness', 'weakness', 'listlessness', 'reduced activity', 'lack of vitality', 'anergia', 'asthenia'],
        
        'loss_of_appetite': ['poor appetite', 'decreased appetite', 'no hunger', 'reduced appetite', 'appetite loss', 'not eating', 'lack of hunger', 'diminished appetite', 'no desire to eat', 'reduced food intake', 'anorexia', 'hyporexia'],
        
        'loss_of_balance': ['unsteady', 'poor balance', 'difficulty balancing', 'balance problems', 'unstable walking', 'coordination problems', 'unsteady gait', 'balance difficulty', 'poor coordination', 'instability', 'ataxia', 'disequilibrium'],
        
        'loss_of_smell': ['no smell', 'cant smell', "can't smell", 'reduced smell', 'smell problems', 'decreased smell', 'impaired smell', 'smell loss', 'smell dysfunction', 'lack of smell', 'anosmia', 'hyposmia'],
        
        'malaise': ['general discomfort', 'feeling unwell', 'not feeling well', 'general illness', 'feeling sick', 'unwellness', 'feeling off', 'general weakness', 'body discomfort', 'illness feeling', 'systemic illness', 'general malaise'],
        
        'mild_fever': ['low fever', 'slight fever', 'low-grade fever', 'mild temperature', 'slight temperature', 'mild pyrexia', 'low temperature', 'slight pyrexia', 'subfebrile', 'mild hyperthermia', 'low-grade pyrexia', 'mild febrile state'],
        
        'mood_swings': ['emotional changes', 'mood changes', 'emotional swings', 'mood fluctuations', 'emotional instability', 'changing moods', 'mood shifts', 'emotional ups and downs', 'mood instability', 'emotional fluctuations', 'cyclothymia', 'emotional lability'],
        
        'movement_stiffness': ['stiff movement', 'rigid movement', 'difficulty moving', 'reduced mobility', 'movement difficulty', 'stiff joints', 'limited movement', 'restricted movement', 'movement problems', 'joint stiffness', 'rigidity', 'motor stiffness'],
        
        'mucoid_sputum': ['mucus in cough', 'phlegm', 'slimy sputum', 'mucus production', 'thick sputum', 'productive cough', 'chest mucus', 'bronchial mucus', 'mucus discharge', 'chest phlegm', 'mucopurulent sputum', 'bronchial secretions'],
        
        'muscle_pain': ['myalgia', 'muscle ache', 'muscular pain', 'sore muscles', 'muscle soreness', 'muscle tenderness', 'muscular ache', 'muscle discomfort', 'painful muscles', 'muscle inflammation', 'fibromyalgia', 'muscle strain'],
        
        'muscle_wasting': ['muscle loss', 'muscle atrophy', 'muscle weakness', 'decreased muscle mass', 'muscle deterioration', 'muscle shrinkage', 'reduced muscle', 'muscle degeneration', 'muscle volume loss', 'muscle breakdown', 'sarcopenia', 'muscular atrophy'],
        
        'muscle_weakness': ['weak muscles', 'muscle fatigue', 'loss of strength', 'decreased muscle strength', 'muscle feebleness', 'reduced muscle power', 'muscle debility', 'muscular weakness', 'weak strength', 'decreased strength', 'myasthenia', 'muscular debility'],
        
        'nausea': ['feeling sick', 'queasiness', 'sick to stomach', 'upset stomach', 'queasy feeling', 'stomach sickness', 'feeling queasy', 'urge to vomit', 'stomach queasiness', 'nauseous feeling', 'emesis', 'gastrointestinal upset'],
        
        'neck_pain': ['cervical pain', 'neck ache', 'sore neck', 'neck stiffness', 'cervical ache', 'neck soreness', 'neck discomfort', 'painful neck', 'neck strain', 'cervical discomfort', 'cervicalgia', 'neck sprain'],
        
        'nodal_skin_eruptions': ['skin nodes', 'skin lumps', 'skin nodules', 'skin bumps', 'raised skin lesions', 'skin growths', 'nodular rash', 'skin eruptions', 'nodular skin lesions', 'skin masses', 'cutaneous nodules', 'dermal nodules'],
        
        'obesity': ['overweight', 'excess weight', 'high body weight', 'weight excess', 'heavy weight', 'excessive fat', 'increased body mass', 'high BMI', 'weight problem', 'excess body fat', 'adiposity', 'corpulence'],
        
        'pain_behind_the_eyes': ['eye socket pain', 'retro-orbital pain', 'orbital pain', 'eye pain', 'eye area pain', 'ocular pain', 'periorbital pain', 'eye region pain', 'pain around eyes', 'eye pressure pain', 'retroorbital pain', 'ophthalmalgia'],
        
        'pain_during_bowel_movements': ['painful bowel movements', 'painful defecation', 'pain when pooping', 'bowel pain', 'painful stool', 'defecation pain', 'painful passing stool', 'rectal pain during bowel movement', 'anal pain during defecation', 'painful elimination', 'proctalgia', 'defecation discomfort'],
        
        'pain_in_anal_region': ['anal pain', 'rectal pain', 'anal discomfort', 'rectal discomfort', 'anal area pain', 'perianal pain', 'anal soreness', 'rectal area pain', 'bottom pain', 'anal region discomfort', 'proctalgia', 'perianal discomfort'],
        
        'painful_walking': ['difficulty walking', 'walking pain', 'gait pain', 'pain while walking', 'walking difficulty', 'painful gait', 'ambulatory pain', 'pain on walking', 'walking discomfort', 'painful mobility', 'dysbasia', 'painful ambulation'],
        
        'palpitations': ['racing heart', 'heart racing', 'rapid heartbeat', 'heart pounding', 'irregular heartbeat', 'fast heartbeat', 'heart fluttering', 'skipped heartbeat', 'rapid pulse', 'heart flutter', 'tachycardia', 'cardiac palpitations'],
        
        'passage_of_gases': ['gas', 'flatulence', 'passing wind', 'intestinal gas', 'excess gas', 'gas passing', 'flatus', 'breaking wind', 'wind', 'gas problems', 'meteorism', 'eructation'],
        
        'patches_in_throat': ['throat spots', 'throat patches', 'throat lesions', 'white patches in throat', 'throat discoloration', 'throat marks', 'pharyngeal patches', 'throat plaques', 'throat spots', 'oral patches', 'pharyngeal lesions', 'tonsillar patches'],
        
        'phlegm': ['mucus', 'sputum', 'chest congestion', 'bronchial secretions', 'chest phlegm', 'mucus production', 'productive cough', 'chest mucus', 'respiratory mucus', 'cough with phlegm', 'bronchial mucus', 'expectoration'],
        
        'polyuria': ['frequent urination', 'excessive urination', 'increased urination', 'frequent peeing', 'excessive peeing', 'urinary frequency', 'frequent bathroom trips', 'increased urine output', 'excess urine production', 'frequent micturition', 'diuresis', 'pollakiuria'],
        
        'prominent_veins_on_calf': ['varicose veins', 'bulging veins', 'visible leg veins', 'swollen leg veins', 'enlarged leg veins', 'protruding veins', 'dilated leg veins', 'visible calf veins', 'leg vein swelling', 'venous prominence', 'phlebectasia', 'venous distention'],
        
        'puffy_face_and_eyes': ['facial swelling', 'swollen face', 'eye swelling', 'face puffiness', 'swollen eyelids', 'facial edema', 'periorbital swelling', 'face bloating', 'puffy eyes', 'facial puffiness', 'facial fullness', 'periorbital edema'],
        
        'pus_filled_pimples': ['pustules', 'acne pustules', 'infected pimples', 'purulent pimples', 'pimples with pus', 'white head pimples', 'infected acne', 'suppurating pimples', 'purulent acne', 'pus spots', 'pyogenic pimples', 'suppurative acne'],
        
        'receiving_blood_transfusion': ['blood transfusion', 'transfusion history', 'previous transfusion', 'blood product transfusion', 'transfusion therapy', 'blood replacement', 'blood product therapy', 'transfusion treatment', 'blood administration', 'blood product administration', 'hemotherapy', 'transfusion procedure'],
        
        'receiving_unsterile_injections': ['unsafe injections', 'dirty needles', 'contaminated injections', 'unclean injections', 'non-sterile shots', 'unsafe needle use', 'contaminated needle use', 'risky injections', 'unsanitary injections', 'unsafe needle practices', 'septic injections', 'contaminated needle exposure'],
        
        'red_sore_around_nose': ['nasal sores', 'nose irritation', 'perinasal redness', 'nose inflammation', 'red nose area', 'nasal skin irritation', 'nose sores', 'red nose sores', 'nasal area redness', 'perinasal inflammation', 'nasal erythema', 'perinasal lesions'],
        
        'red_spots_over_body': ['skin spots', 'red marks', 'body spots', 'red skin spots', 'skin rash spots', 'red skin marks', 'skin redness', 'red patches', 'red lesions', 'cutaneous spots', 'erythematous spots', 'petechiae'],
        
        'redness_of_eyes': ['red eyes', 'eye redness', 'bloodshot eyes', 'eye inflammation', 'conjunctival redness', 'eye irritation', 'ocular redness', 'pink eye', 'inflamed eyes', 'conjunctival inflammation', 'hyperemia', 'conjunctivitis'],
        
        'restlessness': ['agitation', 'unable to rest', 'cant relax', "can't relax", 'fidgety', 'unease', 'anxiety', 'nervous energy', 'nervous tension', 'inner tension', 'akathisia', 'psychomotor agitation'],
        
        'runny_nose': ['nasal discharge', 'rhinorrhea', 'nose running', 'nasal drip', 'watery nose', 'nasal secretion', 'nose dripping', 'fluid from nose', 'nasal mucus', 'nose leaking', 'coryza', 'nasal drainage'],
        
        'rusty_sputum': ['brown sputum', 'rust colored phlegm', 'brown phlegm', 'discolored sputum', 'blood-stained sputum', 'rust colored mucus', 'brownish mucus', 'hemoptysis', 'rusty mucus', 'blood tinged sputum', 'hemorrhagic sputum', 'bloody sputum'],
        
        'scurring': ['skin scaling', 'skin flaking', 'scaly skin', 'skin peeling', 'skin shedding', 'desquamation', 'scaly patches', 'skin scales', 'flaky skin', 'scaling skin', 'cutaneous scaling', 'skin desquamation'],
        
        'shivering': ['chills', 'trembling', 'rigors', 'cold shakes', 'body shakes', 'shaking', 'quivering', 'tremors', 'cold shivers', 'body trembling', 'rigor', 'horripilation'],
        
        'silver_like_dusting': ['silvery scales', 'silver skin patches', 'metallic skin appearance', 'silver skin flakes', 'silvery skin', 'skin silvering', 'silver colored patches', 'metallic skin patches', 'silver skin dust', 'silver scale patches', 'psoriatic scales', 'silvery lesions'],
        
        'sinus_pressure': ['sinus pain', 'facial pressure', 'sinus congestion', 'sinus fullness', 'sinus discomfort', 'facial pain', 'sinus headache', 'nasal pressure', 'sinus problems', 'sinus tension', 'sinusitis', 'rhinosinusitis'],
        
        'skin_peeling': ['peeling skin', 'skin shedding', 'exfoliating skin', 'skin flaking', 'scaling skin', 'desquamation', 'skin scaling', 'skin exfoliation', 'peeling epidermis', 'skin sloughing', 'epidermal peeling', 'cutaneous exfoliation'],
        
        'skin_rash': ['rash', 'dermatitis', 'skin eruption', 'skin inflammation', 'skin irritation', 'skin outbreak', 'cutaneous rash', 'skin lesions', 'skin spots', 'dermal rash', 'exanthem', 'cutaneous eruption'],
        
        'slurred_speech': ['unclear speech', 'speech problems', 'difficulty speaking', 'impaired speech', 'speech impairment', 'speech difficulty', 'unclear speaking', 'speech disturbance', 'speech disorder', 'dysarthria', 'speech slurring', 'articulation problems'],
        
        'small_dents_in_nails': ['nail pitting', 'nail indentations', 'pitted nails', 'nail depressions', 'nail holes', 'nail dimples', 'nail pocks', 'nail deformities', 'nail marks', 'nail pockmarks', 'ungual pitting', 'nail pittings'],
        
        'spinning_movements': ['vertigo', 'dizziness', 'room spinning', 'spinning sensation', 'feeling of spinning', 'rotational vertigo', 'spinning head', 'whirling sensation', 'spinning feeling', 'head spinning', 'giddiness', 'rotatory vertigo'],
        
        'spotting_urination': ['blood in urine', 'urinary blood', 'bloody urine', 'blood spots in urine', 'urinary bleeding', 'pink urine', 'red urine', 'blood stained urine', 'hematuria', 'bloody urination', 'urine blood', 'urinary blood spots'],
        
        'stiff_neck': ['neck stiffness', 'rigid neck', 'limited neck movement', 'neck tension', 'neck tightness', 'cervical rigidity', 'neck mobility problems', 'neck movement difficulty', 'neck strain', 'cervical stiffness', 'torticollis', 'nuchal rigidity'],
        
        'stomach_bleeding': ['gastrointestinal bleeding', 'internal bleeding', 'bleeding ulcer', 'gastric bleeding', 'bloody vomit', 'blood in stool', 'GI bleed', 'hematemesis', 'melena', 'intestinal bleeding', 'gastric hemorrhage', 'gastrointestinal hemorrhage'],
        
        'stomach_pain': ['abdominal pain', 'belly pain', 'tummy ache', 'gastric pain', 'abdominal ache', 'belly ache', 'stomach ache', 'abdominal discomfort', 'gastric discomfort', 'stomach cramps', 'epigastric pain', 'gastrodynia'],
        
        'sunken_eyes': ['deep set eyes', 'hollow eyes', 'eye socket recession', 'orbital depression', 'deep eyes', 'recessed eyes', 'eye hollowing', 'eye depression', 'orbital sunkenness', 'enophthalmos', 'eye socket hollowing', 'ocular depression'],
        
        'sweating': ['perspiration', 'excessive sweating', 'heavy sweating', 'night sweats', 'profuse sweating', 'increased sweating', 'hyperhidrosis', 'diaphoresis', 'sweating episodes', 'excessive perspiration', 'sudoresis', 'hidrosis'],
        
        'swelled_lymph_nodes': ['swollen lymph nodes', 'enlarged lymph nodes', 'lymph node swelling', 'lymphadenopathy', 'swollen glands', 'lymph gland swelling', 'lymphatic swelling', 'lymph enlargement', 'glandular swelling', 'adenopathy', 'lymph swelling', 'lymph node enlargement'],
        
        'swelling_joints': ['joint swelling', 'swollen joints', 'joint inflammation', 'articular swelling', 'inflamed joints', 'joint edema', 'joint enlargement', 'arthritis', 'swollen articulations', 'joint effusion', 'articular edema', 'joint inflammation'],
        
        'swelling_of_stomach': ['abdominal swelling', 'stomach distention', 'bloated stomach', 'abdominal distention', 'stomach bloating', 'belly swelling', 'abdominal enlargement', 'swollen abdomen', 'stomach enlargement', 'abdominal bloating', 'gastric distention', 'ventral swelling'],
        
        'swollen_blood_vessels': ['varicose veins', 'enlarged veins', 'vein swelling', 'dilated blood vessels', 'prominent veins', 'vascular swelling', 'venous swelling', 'swollen veins', 'distended veins', 'phlebitis', 'vascular distention', 'venous distention'],
        
        'swollen_extremeties': ['swollen limbs', 'limb swelling', 'extremity swelling', 'swollen arms and legs', 'peripheral edema', 'limb edema', 'swelling of limbs', 'extremity edema', 'peripheral swelling', 'edema of extremities', 'appendicular edema', 'limb swelling'],
        
        'swollen_legs': ['leg swelling', 'edema in legs', 'swollen lower limbs', 'leg edema', 'lower extremity swelling', 'puffy legs', 'leg inflammation', 'lower limb swelling', 'pedal edema', 'leg enlargement', 'lower extremity edema', 'leg puffiness'],
        
        'throat_irritation': ['sore throat', 'throat pain', 'throat discomfort', 'pharyngeal irritation', 'throat soreness', 'pharyngitis', 'throat inflammation', 'throat ache', 'painful throat', 'pharyngeal pain', 'throat burning', 'pharyngeal discomfort'],
        
        'toxic_look_(typhos)': ['toxic appearance', 'sick appearance', 'ill appearance', 'unhealthy look', 'toxic facies', 'septic appearance', 'ill looking', 'toxic looking', 'sick looking', 'morbid appearance', 'toxic syndrome', 'typhoid facies'],
        
        'ulcers_on_tongue': ['tongue sores', 'tongue ulceration', 'mouth ulcers', 'tongue lesions', 'oral ulcers', 'lingual ulcers', 'tongue sores', 'mouth sores', 'oral lesions', 'lingual sores', 'glossal ulcers', 'stomatitis'],
        
        'unsteadiness': ['poor balance', 'instability', 'lack of coordination', 'wobbliness', 'unstable gait', 'balance problems', 'coordination problems', 'unsteady movement', 'loss of balance', 'ataxia', 'gait instability', 'postural instability'],
        
        'visual_disturbances': ['vision problems', 'sight disturbance', 'visual problems', 'disturbed vision', 'vision changes', 'visual changes', 'sight problems', 'visual difficulties', 'vision distortion', 'visual impairment', 'vision abnormalities', 'visual dysfunction'],
        
        'vomiting': ['throwing up', 'nausea with vomiting', 'being sick', 'emesis', 'regurgitation', 'retching', 'getting sick', 'stomach emptying', 'gastric emptying', 'puking', 'hyperemesis', 'vomitus'],
        
        'watering_from_eyes': ['teary eyes', 'excessive tearing', 'eye watering', 'watery eyes', 'tear overflow', 'excessive tears', 'eye tearing', 'lacrimation', 'epiphora', 'tearing eyes', 'eye discharge', 'tear production'],
        
        'weakness_in_limbs': ['limb weakness', 'weak arms and legs', 'muscle weakness', 'extremity weakness', 'weak limbs', 'loss of strength', 'weak muscles', 'reduced limb strength', 'motor weakness', 'paresis', 'limb paresis', 'muscular weakness'],
        
        'weakness_of_one_body_side': ['hemiparesis', 'one sided weakness', 'unilateral weakness', 'half body weakness', 'partial paralysis', 'one side weakness', 'hemiplegic weakness', 'lateral weakness', 'unilateral paresis', 'hemiplegia', 'one sided paralysis', 'unilateral paralysis'],
        
        'weight_gain': ['increased weight', 'gaining weight', 'weight increase', 'added weight', 'weight growth', 'increased body weight', 'weight addition', 'putting on weight', 'added pounds', 'weight accumulation', 'obesity', 'adiposity'],
        
        'weight_loss': ['losing weight', 'decreased weight', 'weight decrease', 'weight reduction', 'reduced weight', 'weight dropping', 'losing pounds', 'weight dropping', 'slimming', 'weight reduction', 'emaciation', 'cachexia'],
        
        'yellow_crust_ooze': ['yellow discharge', 'crusty discharge', 'yellow scabs', 'yellow exudate', 'crusty exudate', 'yellow crusting', 'crusty oozing', 'yellow secretion', 'purulent discharge', 'yellow pus', 'yellow drainage', 'crusty secretion'],
        
        'yellow_urine': ['dark urine', 'amber urine', 'golden urine', 'colored urine', 'urine discoloration', 'darkened urine', 'amber colored urine', 'yellow colored urine', 'urine color change', 'concentrated urine', 'choluria', 'xanthuria'],
        
        'yellowing_of_eyes': ['jaundiced eyes', 'yellow sclera', 'eye yellowing', 'yellow eye whites', 'icteric eyes', 'yellow discoloration of eyes', 'eye jaundice', 'scleral icterus', 'yellow sclera', 'icteric sclera', 'ocular jaundice', 'eye icterus'],
        
        'yellowish_skin': ['jaundice', 'yellow skin', 'skin yellowing', 'icteric skin', 'yellow complexion', 'skin discoloration', 'yellow discoloration', 'yellow tint', 'yellow skin tone', 'icterus', 'xanthosis', 'yellow pigmentation']
    }
        
    # Create reverse lookup for quick matching
        self.symptom_lookup = {}
        for main_symptom, variations in self.symptom_variations.items():
            for variation in variations:
                self.symptom_lookup[variation.lower()] = main_symptom
            # Also add the main symptom itself
            self.symptom_lookup[main_symptom.replace('_', ' ').lower()] = main_symptom

    

    def _get_phrases(self, text: str) -> List[str]:
        """Extract potential symptom phrases from text."""
        # Ensure we're working with a string
        if isinstance(text, list):
            text = ' '.join(text)
        text = str(text).lower()
        
        # Split into words
        words = text.split()
        phrases = []
        
        # Generate phrases of different lengths
        for i in range(len(words)):
            for j in range(1, 6):  # Check phrases up to 5 words long
                if i + j <= len(words):
                    phrase = ' '.join(words[i:i+j])
                    phrases.append(phrase)
                    
        return phrases

    def find_matching_symptoms(self, text: str) -> List[str]:
        """
        Find all matching symptoms in the text using various matching techniques.
        Returns list of matched symptom names.
        """
        text = str(text).lower().strip()
        matched_symptoms = set()

        # 1. Direct matches from variations dictionary
        phrases = self._get_phrases(text)  # Pass the text directly, not words
        
        for phrase in phrases:
            if phrase in self.symptom_lookup:
                matched_symptoms.add(self.symptom_lookup[phrase])

        # 2. Fuzzy matching for near matches
        for symptom in self.symptoms:
            if symptom not in matched_symptoms:  # Only check if not already matched
                clean_symptom = symptom.replace('_', ' ')
                variations = self.symptom_variations.get(symptom, [])
                
                # Check main symptom
                if self._fuzzy_match(clean_symptom, text):
                    matched_symptoms.add(symptom)
                    continue
                    
                # Check variations
                for variation in variations:
                    if self._fuzzy_match(variation, text):
                        matched_symptoms.add(symptom)
                        break

        return list(matched_symptoms)

    def _fuzzy_match(self, pattern: str, text: str, threshold: float = 0.85) -> bool:
        """Perform fuzzy matching between pattern and text."""
        # Ensure we're working with strings
        pattern = str(pattern).lower()
        text = str(text).lower()
        
        # Direct substring check first
        if pattern in text:
            return True
            
        # Check each word/phrase combination
        phrases = self._get_phrases(text)  # Pass text directly
        
        for phrase in phrases:
            similarity = difflib.SequenceMatcher(None, pattern, phrase).ratio()
            if similarity >= threshold:
                return True
                
        return False
            
    def detect_input_type(self, text: str) -> str:
        """
        Determine if input is symptoms or question based on symptom count.
        Returns 'symptoms' if enough symptoms are detected, 'qa' otherwise.
        """
        matched_symptoms = self.find_matching_symptoms(text)
        return "symptoms" if len(matched_symptoms) >= 4 else "qa"
    # def find_matching_symptoms(self, text: str) -> List[str]:
    #     """
    #     Find all matching symptoms in the text using various matching techniques.
    #     Returns list of matched symptom names.
    #     """
    #     text = text.lower().strip()
    #     matched_symptoms = set()

    #     # 1. Direct matches from variations dictionary
    #     words = text.split()
    #     phrases = self._get_phrases(words)
        
    #     for phrase in phrases:
    #         if phrase in self.symptom_lookup:
    #             matched_symptoms.add(self.symptom_lookup[phrase])

    #     # 2. Fuzzy matching for near matches
    #     for symptom in self.symptoms:
    #         if symptom not in matched_symptoms:  # Only check if not already matched
    #             clean_symptom = symptom.replace('_', ' ')
    #             variations = self.symptom_variations.get(symptom, [])
                
    #             # Check main symptom
    #             if self._fuzzy_match(clean_symptom, text):
    #                 matched_symptoms.add(symptom)
    #                 continue
                    
    #             # Check variations
    #             for variation in variations:
    #                 if self._fuzzy_match(variation, text):
    #                     matched_symptoms.add(symptom)
    #                     break

    #     return list(matched_symptoms)
    def process_symptoms(self, text: str) -> str:
        """
        Process symptoms text and return prediction with detailed feedback.
        """
        try:
            # Find all matching symptoms
            matched_symptoms = self.find_matching_symptoms(text)
            symptom_count = len(matched_symptoms)
            
            # Prepare feature vector
            features = np.zeros(len(self.symptoms))
            for symptom in matched_symptoms:
                if symptom in self.symptoms:
                    idx = self.symptoms.index(symptom)
                    features[idx] = 1
            
            # Format detected symptoms for output
            detected_symptoms = [s.replace('_', ' ').title() for s in matched_symptoms]
            symptoms_str = ', '.join(detected_symptoms)
            
            # If not enough symptoms, provide guidance
            if symptom_count < 4:
                return (
                    f"I detected these symptoms: {symptoms_str}\n"
                    f"Current symptom count: {symptom_count}\n"
                    f"Please provide at least 4 symptoms for an accurate prediction.\n"
                    f"Try describing your symptoms in more detail or mention additional symptoms you may have."
                )
            
            # Make prediction if enough symptoms
            # prediction = self.rf_model.predict([features])[0]
            numeric_prediction = self.rf_model.predict([features])[0]
            prediction = self.label_encoder.inverse_transform([numeric_prediction])[0]
            
            return (
                f"Detected Symptoms ({symptom_count}):\n"
                f"{symptoms_str}\n\n"
                f"Based on these symptoms, you might have: {prediction}\n\n"
                "Note: This is not a definitive diagnosis. Please consult a healthcare professional for proper medical evaluation."
            )
            
        except Exception as e:
            print(f"Error processing symptoms: {e}")
            return "An error occurred while processing your symptoms. Please try again."

    def process_qa(self, question: str) -> str:
        """Process medical question using T5 model."""
        try:
            # Check if it might be symptoms described as a question
            matched_symptoms = self.find_matching_symptoms(question)
            if len(matched_symptoms) >= 4:
                return self.process_symptoms(question)
                
            # Otherwise process as a medical question
            input_text = f"answer medical: {question}"
            answer = self.qa_model.predict(input_text)[0]
            
            return answer.strip() if answer.strip() else "Could not find a specific answer to your medical question."
            
        except Exception as e:
            print(f"Error in QA processing: {e}")
            return "An error occurred while processing your question. Please try again."
    def speech_to_text(self, audio_bytes: bytes, lang: str = 'en') -> str:
        """Convert speech to text with language support."""
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
            try:
                # Specify language for speech recognition
                return recognizer.recognize_google(audio, language=lang)
            except sr.UnknownValueError:
                return "Speech not understood"
            except sr.RequestError:
                return "Could not process speech"

    def text_to_speech(self, text: str, lang: str = 'en') -> bytes:
        """Convert text to speech."""
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()

    

    # def translate_text(self, text: str, target_lang: str) -> str:
    #     """Translate text to target language."""
    #     try:
    #         if target_lang != 'en':
    #             # GoogleTranslator.translate() returns a string directly
    #             return self.translator.translate(text)
    #         return text
    #     except Exception as e:
    #         print(f"Translation error: {e}")
    #         return text  
def main():
    st.title("Medical Chatbot")
    local_css()  # Apply your custom CSS
    
    # Initialize chatbot
    chatbot = MedicalChatbot()
    
    # Initialize session state for recorder
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = LiveAudioRecorder()
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    
    # Language selection
    languages = {
        'English': 'en',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
        'Chinese': 'zh-cn',
        'Hindi': 'hi',
        'Tamil': 'ta',
        'Telugu': 'te'
    }
    selected_lang = st.selectbox("Select Language", list(languages.keys()))
    
    # Input method selection
    input_method = st.radio("Choose input method", ["Text", "Voice Upload", "Live Recording"])
    
    if input_method == "Text":
        user_input = st.text_input("Enter your question or symptoms:")
        if st.button("Submit"):
            if user_input:
                process_input(user_input, selected_lang, chatbot, languages)
    
    elif input_method == "Voice Upload":
        # Existing voice upload code
        audio_file = st.file_uploader("Upload audio", type=['wav', 'mp3'])
        if audio_file is not None:
            process_audio_file(audio_file, selected_lang, chatbot, languages)
    
    else:  # Live Recording
        col1, col2 = st.columns(2)
        
        # Recording controls
        with col1:
            if not st.session_state.recording:
                if st.button("Start Recording", key="start"):
                    st.session_state.recording = True
                    st.session_state.audio_recorder.start_recording()
                    st.rerun()
            else:
                if st.button("Stop Recording", key="stop"):
                    st.session_state.recording = False
                    audio_data = st.session_state.audio_recorder.stop_recording()
                    if audio_data:
                        # Process the recorded audio
                        try:
                            # Display audio player
                            st.audio(audio_data, format='audio/wav')
                            
                            # Convert speech to text
                            recognizer = sr.Recognizer()
                            with sr.AudioFile(io.BytesIO(audio_data)) as source:
                                audio = recognizer.record(source)
                                user_input = recognizer.recognize_google(
                                    audio, 
                                    language=languages[selected_lang]
                                )
                                
                                st.write(f"You said: {user_input}")
                                process_input(user_input, selected_lang, chatbot, languages)
                                
                        except sr.UnknownValueError:
                            st.error("Could not understand the audio. Please try again.")
                        except sr.RequestError as e:
                            st.error(f"Could not process speech: {str(e)}")
                        except Exception as e:
                            st.error(f"Error processing audio: {str(e)}")
                    
        
        # Recording status
        with col2:
            if st.session_state.recording:
                st.markdown("🔴 Recording...")
                # Add a progress bar or animation while recording
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.1)
                    progress_bar.progress(i + 1)
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; padding: 20px; background: rgba(255,255,255,0.9); border-radius: 15px;'>
        <p>👨‍⚕️ Your AI Medical Assistant 🤖</p>
        <p>⚕️ Available 24/7 for Your Health Queries 🏥</p>
        <small>🔒 Private & Secure | 🌐 Multi-language Support | 🎯 Accurate Predictions</small>
    </div>
    """, unsafe_allow_html=True)

def process_input(user_input: str, selected_lang: str, chatbot: MedicalChatbot, languages: dict):
    """Process user input and generate response with proper translation flow"""
    try:
        # First translate input to English if not in English
        if selected_lang != "English":
            print(languages[selected_lang])
            translator = GoogleTranslator(source=languages[selected_lang], target='en')
            translated_input = translator.translate(user_input)
            print(f"Translated to English: {translated_input}")
            print('user_input')
            print(user_input)
            
            
        else:
            translated_input = user_input
        
        # Now detect input type using translated text
        input_type = chatbot.detect_input_type(translated_input)
        print(f"Detected input type: {input_type}")  # Debug print
        
        # Process the translated input
        if input_type == "symptoms":
            response = chatbot.process_symptoms(translated_input)
        else:
            response = chatbot.process_qa(translated_input)
        
        # Translate response back to original language if needed
        if selected_lang != "English":
            translator = GoogleTranslator(source='en', target=languages[selected_lang])
            response = translator.translate(response)
        
        st.write(response)
        
        # Generate audio response
        response_audio = chatbot.text_to_speech(response, languages[selected_lang])
        st.audio(response_audio, format='audio/mp3')
    
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        st.write("Please try again or contact support if the issue persists.")

def process_audio_file(audio_file, selected_lang: str, chatbot: MedicalChatbot, languages: dict):
    """Process uploaded audio file"""
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
    
    try:
        user_input = chatbot.speech_to_text(audio_bytes, languages[selected_lang])
        st.write(f"You said: {user_input}")
        
        if user_input and user_input not in ["Speech not understood", "Could not process speech"]:
            process_input(user_input, selected_lang, chatbot, languages)
        else:
            st.error("Could not understand the audio. Please try again.")
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.write("Please try again or use text input instead.")

if __name__ == "__main__":
    main()