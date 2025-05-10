import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
import scispacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.utils import to_categorical
import warnings
import pickle
import string
import re
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
def read_data(path_to_csv_file=None, delimiter=None):
    '''
    Reads csv file from specified path or default location
    '''
    default_path = '/app/patient_behavior_data.csv'  
    path = path_to_csv_file if path_to_csv_file else default_path
    df = pd.read_csv(path, delimiter=delimiter)
    return df

df_patient = read_data(delimiter=";")
df_patient.head()

# %%
# Replace "None" records with "Not Specified"
def replace_missing_with_none(df, column_names):
    '''
    Replace missing values in the specified columns with None.
    '''

    for column_name in column_names:
        if column_name in df.columns:
            df[column_name] = df[column_name].where(df[column_name].notna(), "Not Specified")
        else:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    return df

patient_df_cleaned = replace_missing_with_none(df_patient, ['medication'])
patient_df_cleaned.head()

# %%
# Now extract the numerical value from dose and fill the missing values with median. eg 15mg will be 15
def clean_and_fill_dose(df, dose_column='dose'):
    def extract_dose_value(dose):
        if pd.isna(dose):
            return None
        return float(dose.rstrip('mg'))

    df['dose'] = df[dose_column].apply(extract_dose_value)

    # fill in missing dose values with median
    median_dose = df['dose'].median()
    df['dose'].fillna(median_dose, inplace=True)

    return df

patient_df_cleaned = clean_and_fill_dose(patient_df_cleaned)
patient_df_cleaned.head()

# %%
def clean_clinical_notes(text):
    """
    Keep only the first coherent sentence from doctor notes.
    Removes all text after the first period followed by space and capital letter.
    """
    if pd.isna(text):
        return ""

    # Find the first sentence-ending period followed by space and capital letter
    match = re.search(r'\.\s+[A-Z]', str(text))

    if match:
        return text[:match.start()+1].strip()
    else:
        return str(text).strip()

patient_df_cleaned['doctor_notes'] = patient_df_cleaned['doctor_notes'].apply(clean_clinical_notes)
patient_df_cleaned.head()

# %%
# Load spaCy's English model
nlp_spacy = spacy.load('en_core_web_sm')

# Keep some custom medical terms stopwords
MEDICAL_STOPWORDS = {'patient', 'history', 'normal', 'exam', 'physical', 'day', 'week', 'month', 'year', 'status', 'note', 'findings'}
STOP_WORDS.update(MEDICAL_STOPWORDS)

def remove_emojis_special_chars(text):
    """Remove emojis and special characters from text"""
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002500-\U00002BEF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"
                           u"\u3030"
                           "]+", flags=re.UNICODE)

    text = re.sub(r'[^\w\s.,!?]', '', text)
    return emoji_pattern.sub(r'', text)

def preprocess_text(text):
    """
    Preprocess clinical notes by:
    1. Removing emojis/special characters
    2. Tokenizing
    3. Lemmatizing
    4. Removing stopwords (while preserving medical terms)
    """
    if pd.isna(text):
        return ""

    # lower case text
    cleaned_text = remove_emojis_special_chars(str(text).lower())

    # tokenization
    doc = nlp_spacy(cleaned_text)

    processed_tokens = []
    for token in doc:
        if (not token.is_punct and
            not token.is_space and
            (not token.is_stop or token.text in MEDICAL_STOPWORDS)):

            # lemmatization
            lemma = token.lemma_.strip().lower()
            if lemma:
                processed_tokens.append(lemma)

    return " ".join(processed_tokens)

patient_df_processed = patient_df_cleaned.copy()
patient_df_processed['processed_notes'] = patient_df_processed['doctor_notes'].apply(preprocess_text)
patient_df_processed.head()

# %% [markdown]
# #### 1. Prepare features: encode categorical and normalise numeric.

# %%
# perfome one-hot encoding on gender and medicaiton
def one_hot_encode_columns(df, columns_to_encode):
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=columns_to_encode)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded

columns_to_encode = ['gender', 'medication']
encoded_df = one_hot_encode_columns(patient_df_processed, columns_to_encode)
encoded_df.head()

# %%
def normalize_features(df, numerical_features):
    """
    Apply appropriate normalization technique to each feature.
    """
    # Create a copy of the dataframe to avoid modifying the original
    normalized_df = df.copy()

    # Initialize scalers
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # BMI - Log transform + MinMax scaling (right-skewed)
    normalized_df['bmi_normalized'] = df['bmi'].map(lambda x: np.log1p(x) if x > 0 else 0)
    normalized_df['bmi_normalized'] = minmax_scaler.fit_transform(normalized_df[['bmi_normalized']])

    # Weight - Standard scaling (uniform distribution)
    normalized_df['weight_normalized'] = std_scaler.fit_transform(df[['weight']])

    # Height - Standard scaling
    normalized_df['height_normalized'] = std_scaler.fit_transform(df[['height']])

    # Systolic BP - Standard scaling
    normalized_df['systolic_normalized'] = std_scaler.fit_transform(df[['systolic']])

    # Diastolic BP - Standard scaling
    normalized_df['diastolic_normalized'] = std_scaler.fit_transform(df[['diastolic']])

    # Dose - Log transform + MinMax scaling (right-skewed with multiple peaks)
    normalized_df['dose_normalized'] = df['dose'].map(lambda x: np.log1p(x) if x > 0 else 0)
    normalized_df['dose_normalized'] = minmax_scaler.fit_transform(normalized_df[['dose_normalized']])

    # view before and after normalization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['bmi'], bins=30)
    plt.title('Original BMI')
    plt.subplot(1, 2, 2)
    plt.hist(normalized_df['bmi_normalized'], bins=30)
    plt.title('Normalized BMI')
    plt.tight_layout()
    plt.show()

    return normalized_df

numerical_features = ['bmi', 'weight', 'height', 'systolic', 'diastolic', 'dose']
normalized_df = normalize_features(encoded_df, numerical_features)
normalized_df.head()

# %% [markdown]
# #### 2. Baseline: Predict concentration using Random Forest.

# %%
def set_features_and_target(df):
    '''
    Returns two data frames with features and target variables.
    '''
    X = df.drop(['patient_id','name','surname','dose','bmi','weight','height',
                  'systolic','diastolic','doctor_notes','processed_notes','concentration'], axis=1)
    y = df['concentration']

    return X,y

X,y = set_features_and_target(normalized_df)
print(X.shape, y.shape)

# %%
def train_test_split_df(X,y):
    '''
    Creates train and test split.
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_df(X,y)
print(f"\n=== Data Split - concentration prediction ===")
print("X_train, y_train shape:", X_train.shape,y_train.shape)
print("X_test, y_test shape:", X_test.shape,y_test.shape)

# %%
def model_application(X_train,y_train,optimiser):
    '''
    Model application. If optimiser is true , a grid search is applied to optimise the model.
    '''
    if optimiser == True:
                params = {
                'max_features': [1, 3, 10],
                'min_samples_split': [2, 3, 10],
                'min_samples_leaf': [1, 3, 10],
                'criterion': ["entropy", "gini"]
                }

                cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=15)

                # Building model
                rf = RandomForestClassifier()
                grid = GridSearchCV(rf, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)

                # Fitting the model
                grid.fit(X_train, y_train)

                dtc_grid_val_score = grid.best_score_
                print('Best Score:', dtc_grid_val_score)
                print('Best Params:', grid.best_params_)
                print('Best Estimator:', grid.best_estimator_)

                rf_model = grid.best_estimator_
    else:
                rf_model = RandomForestClassifier(n_estimators=100,
                                                    max_depth=5,
                                                    min_samples_split=10,
                                                    min_samples_leaf=8,
                                                    max_features='sqrt',
                                                    bootstrap=True,
                                                    oob_score=True,
                                                    class_weight='balanced',
                                                    random_state=42
                                                  )
                rf_model.fit(X_train, y_train)

    return rf_model

rf_model = model_application(X_train,y_train,optimiser=False)

# %%
def model_predict(X_test):
    '''
    Create y_pred , model prediction based on test set features.
    '''
    y_pred = rf_model.predict(X_test)

    return y_pred

y_pred = model_predict(X_test)
print('Training Accuracy Score - concentration pred (%):',rf_model.score(X_train,y_train)*100)
print('Test Accuracy Score - concentration pred (%):',rf_model.score(X_test,y_test)*100)

# %%
def generate_performance_metrics(y_test, y_pred, model_name=""):
    score = accuracy_score(y_test, y_pred)
    if model_name:
        print(f"\n=== {model_name} Performance - concentration prediction ===")
    print('Model Accuracy:', score)
    print('Classification Report - concentration prediction :\n', classification_report(y_test, y_pred, zero_division=0))

    return score

y_pred_rf = rf_model.predict(X_test)
generate_performance_metrics(y_test, y_pred_rf, "Random Forest")

# %% [markdown]
# #### 3. LSTM: Predict impulsivity from doctor_notes.

# %%
# def lstm_model(patient_df_processed):
#     # Preprocess labels (-2 to 2) -> (0 to 4)
#     X = patient_df_processed['processed_notes'].values
#     y = patient_df_processed['impulsivity'].values + 2

#     # Split data (stratified to maintain class balance)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )

#     # Text tokenization with limited vocabulary
#     tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
#     tokenizer.fit_on_texts(X_train)

#     # Convert text to sequences and pad
#     max_length = 30
#     X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length)
#     X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_length)

#     # Model architecture
#     model = Sequential([
#         Embedding(1000, 16, input_length=max_length),
#         LSTM(32, dropout=0.1),
#         Dense(5, activation='softmax')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     # Training
#     print("Training model...")
#     history = model.fit(
#         X_train, y_train,
#         epochs=5,
#         batch_size=64,
#         validation_split=0.2,
#         verbose=1
#     )

#     # Training metrics
#     print("\nTraining Accuracy per Epoch:")
#     for epoch, acc in enumerate(history.history['accuracy'], 1):
#         print(f"Epoch {epoch}: {acc:.4f}")

#     # Evaluation
#     print("\nEvaluating on test set...")
#     test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

#     # Predictions
#     y_pred = model.predict(X_test, verbose=0)
#     y_pred_classes = np.argmax(y_pred, axis=1)
#     y_test_original = y_test - 2  # Convert back to original scale (-2 to 2)
#     y_pred_original = y_pred_classes - 2

#     f1 = f1_score(y_test, y_pred_classes, average='weighted')
#     precision = precision_score(y_test, y_pred_classes, average='weighted')
#     recall = recall_score(y_test, y_pred_classes, average='weighted')
#     mae = mean_absolute_error(y_test_original, y_pred_original)

#     # Print metrics
#     print("\n================= Metrics =================")
#     print(f"Test Accuracy: {test_acc:.4f}")
#     print(f"Test Loss: {test_loss:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"MAE: {mae:.4f}")

#     print("\n================ Classification Report ===============")
#     print(classification_report(
#         y_test_original, y_pred_original,
#         target_names=['-2', '-1', '0', '1', '2']
#     ))

#     return model, tokenizer

# model, tokenizer = lstm_model(patient_df_processed)

# %%
# def clstm_model(patient_df_processed):
#     # Separate features and target variable and encode variables
#     X = patient_df_processed.drop(['impulsivity','doctor_notes','patient_id','name','surname'], axis=1)
#     y = patient_df_processed['impulsivity']

#     categorical_cols = ['gender', 'medication']
#     for col in categorical_cols:
#         le = LabelEncoder()
#         X[col] = le.fit_transform(X[col])

#     ordinal_cols = ['concentration', 'distractibility',  'hyperactivity', 'sleep', 'mood', 'appetite']
#     ordinal_mapping = {-2: 0,-1: 1,0: 2, 1: 3, 2: 4}
#     for col in ordinal_cols:
#         X[col] = X[col].map(ordinal_mapping)

#     # Scale numerical features
#     numerical_cols = ['dose', 'bmi', 'weight', 'height', 'systolic', 'diastolic']
#     scaler = StandardScaler()
#     X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

#     # Text processing
#     text_col = 'processed_notes'
#     max_words = 1000
#     tokenizer = Tokenizer(num_words=max_words, oov_token="<oov>")
#     tokenizer.fit_on_texts(X[text_col])
#     sequences = tokenizer.texts_to_sequences(X[text_col])
#     max_sequence_length = 200
#     padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

#     y = to_categorical(y, num_classes=5)

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#     # Separate text and tabular data for the model
#     X_train_text = padded_sequences[X_train.index]
#     X_test_text = padded_sequences[X_test.index]

#     X_train_tabular = X_train.drop(text_col, axis=1)
#     X_test_tabular = X_test.drop(text_col, axis=1)

#     # Model Architecture (CLSTM)
#     text_input = Input(shape=(max_sequence_length,), name='text_input')
#     embedding_layer = Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length)(text_input)
#     conv1d_layer = Conv1D(filters=64, kernel_size=5, activation='relu')(embedding_layer)
#     maxpooling1d_layer = MaxPooling1D(pool_size=4)(conv1d_layer)
#     lstm_layer = LSTM(units=128)(maxpooling1d_layer)
#     dropout_text = Dropout(0.5)(lstm_layer)

#     # Tabular input branch
#     tabular_input = Input(shape=(X_train_tabular.shape[1],), name='tabular_input')
#     dense_tabular = Dense(units=64, activation='relu')(tabular_input)
#     dropout_tabular = Dropout(0.5)(dense_tabular)
#     combined = Concatenate()([dropout_text, dropout_tabular])
#     dense_combined = Dense(units=128, activation='relu')(combined)
#     output = Dense(units=5, activation='softmax')(dense_combined)
#     model = Model(inputs=[text_input, tabular_input], outputs=output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     print(model.summary())

#     # Train the model
#     # Convert tabular data to numpy arrays
#     X_train_tabular_np = np.array(X_train_tabular).astype(np.float32)
#     X_test_tabular_np = np.array(X_test_tabular).astype(np.float32)

#     history = model.fit(
#         [X_train_text, X_train_tabular_np],
#         y_train,
#         epochs=5,
#         batch_size=64,
#         validation_split=0.2,
#         verbose=1
#     )

#     # Evaluate the model
#     y_pred_train = model.predict([X_train_text, X_train_tabular_np])
#     y_pred_test = model.predict([X_test_text, X_test_tabular_np])
#     y_pred_train_classes = np.argmax(y_pred_train, axis=1)
#     y_pred_test_classes = np.argmax(y_pred_test, axis=1)

#     y_train_classes = np.argmax(y_train, axis=1)
#     y_test_classes = np.argmax(y_test, axis=1)

#     # Convert back to original scale (-2 to 2)
#     y_train_original = y_train_classes - 2
#     y_test_original = y_test_classes - 2
#     y_pred_train_original = y_pred_train_classes - 2
#     y_pred_test_original = y_pred_test_classes - 2

#     # Print training and testing accuracy
#     train_accuracy = accuracy_score(y_train_original, y_pred_train_original)
#     test_accuracy = accuracy_score(y_test_original, y_pred_test_original)
#     print(f"Training Accuracy: {train_accuracy:.4f}")
#     print(f"Testing Accuracy: {test_accuracy:.4f}")

#     # Calculate additional metrics
#     train_f1 = f1_score(y_train_original, y_pred_train_original, average='weighted')
#     test_f1 = f1_score(y_test_original, y_pred_test_original, average='weighted')
#     train_precision = precision_score(y_train_original, y_pred_train_original, average='weighted')
#     test_precision = precision_score(y_test_original, y_pred_test_original, average='weighted')
#     train_recall = recall_score(y_train_original, y_pred_train_original, average='weighted')
#     test_recall = recall_score(y_test_original, y_pred_test_original, average='weighted')
#     train_mae = mean_absolute_error(y_train_original, y_pred_train_original)
#     test_mae = mean_absolute_error(y_test_original, y_pred_test_original)

#     print(f"Training F1 Score: {train_f1:.4f}")
#     print(f"Testing F1 Score: {test_f1:.4f}")
#     print(f"Training Precision: {train_precision:.4f}")
#     print(f"Testing Precision: {test_precision:.4f}")
#     print(f"Training Recall: {train_recall:.4f}")
#     print(f"Testing Recall: {test_recall:.4f}")
#     print(f"Training MAE: {train_mae:.4f}")
#     print(f"Testing MAE: {test_mae:.4f}")

#     # Print classification report
#     print("\nClassification Report - Training Data:")
#     print(classification_report(
#         y_train_original, y_pred_train_original,
#         target_names=['-2', '-1', '0', '1', '2']
#     ))
#     print("\nClassification Report - Testing Data:")
#     print(classification_report(
#         y_test_original, y_pred_test_original,
#         target_names=['-2', '-1', '0', '1', '2']
#     ))

#     return model, history, y_test_original, y_pred_test_original

# model, history, y_test_original, y_pred_test_original = clstm_model(patient_df_processed)
