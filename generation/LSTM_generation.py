import os
import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import stream, note, instrument, tempo


# Step 1: Define the Model Architecture (same as during training)
def build_LSTM_model(sequence_length, n_unique_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape=(sequence_length, 5), return_sequences=True))  # Adjusted to 5 input features
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_unique_notes, activation='softmax'))
    return model


# Method that calls the other method that builds the model, then loads the trained weights
def load_LSTM_model(sequence_length, n_unique_notes, checkpoint_path):
    model = build_LSTM_model(sequence_length, n_unique_notes)
    model.load_weights(checkpoint_path)

    return model


# Method to load mappings
def load_mappings(mapping_path=r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend\generation'
                               r'\Subset_Dataset/note_to_int.pkl'):
    with open(mapping_path, 'rb') as f:
        note_to_int = pickle.load(f)
    int_to_note = {number: note for note, number in note_to_int.items()}
    rest_indices = [index for index, note in int_to_note.items() if note.lower() == 'rest' or note == 'R']

    return note_to_int, int_to_note, rest_indices


# Method to choose star sequence
def load_start_sequence():
    with open(r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend\generation\Subset_Dataset'
              r'\input_sequences_notes.pkl', 'rb') as f:
        X_notes = pickle.load(f)
    with open(r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend\generation\Subset_Dataset'
              r'\input_sequences_durations.pkl', 'rb') as f:
        X_durations = pickle.load(f)
    with open(r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend\generation\Subset_Dataset'
              r'\input_sequences_tempos.pkl', 'rb') as f:
        X_tempos = pickle.load(f)
    with open(r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend\generation\Subset_Dataset'
              r'\input_sequences_time_signatures.pkl', 'rb') as f:
        X_time_signatures = pickle.load(f)
    with open(r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend\generation\Subset_Dataset'
              r'\input_sequences_key_signatures.pkl', 'rb') as f:
        X_key_signatures = pickle.load(f)

    # Concatenate all features to create the full start sequence
    X = np.concatenate([X_notes, X_durations, X_tempos, X_time_signatures, X_key_signatures], axis=-1)

    # Choose a random start sequence from the training data
    start_sequence = X[np.random.randint(0, len(X))]

    return start_sequence


def generate_music_lstm(model, start_sequence, int_to_note, rest_indices, valid_scale_notes, durations, n_generate=100,
                        temperature=1.0, lower_range=60, upper_range=70):
    generated_notes = []
    generated_durations = []
    current_sequence = np.reshape(start_sequence,
                                  (1, len(start_sequence), 5))  # Reshape seed sequence to match model input

    # Define C Major scale notes and desired MIDI range
    c_major_notes = ['C', 'D', 'E', 'F', 'G', 'A']  # Adjust if you want to include 'B'
    desired_octaves = [4]

    while len(generated_notes) < n_generate:
        # Predict the next note
        prediction = model.predict(current_sequence, verbose=0)[0]

        # Zero out the probabilities for rests
        prediction[rest_indices] = 0

        # Apply temperature scaling
        prediction = prediction ** (1 / temperature)
        prediction = prediction / np.sum(prediction)  # Renormalize

        # Filter the prediction to only valid notes in C major scale and desired octave
        valid_indices = []
        for idx, note_str in int_to_note.items():
            # Skip chords and invalid patterns
            if '.' in note_str or note_str.isdigit():
                continue
            # Get the note properties
            try:
                n = note.Note(note_str)
                note_name = n.pitch.name  # Note name (e.g., 'C')
                note_midi = n.pitch.midi  # MIDI number
            except:
                continue  # Skip if the note_str cannot be parsed into a note

            # Check if the note is in C major, desired octave, and within the desired MIDI range
            if (
                    note_name in valid_scale_notes and
                    lower_range <= note_midi <= upper_range
            ):
                valid_indices.append(idx)

        # If no valid notes remain, fall back to random snapping or handle accordingly
        if len(valid_indices) == 0:
            print(f"No valid notes in scale and range at position {len(generated_notes)}/{n_generate}.")
            # Optionally, select a random note from C major within the desired octave
            new_valid_note_names = [n + str(o) for n in c_major_notes for o in desired_octaves]
            snapped_note = random.choice(new_valid_note_names)
            print(f"Snapped to random valid note: {snapped_note}")
            generated_notes.append(snapped_note)
            generated_durations.append(1.0 if len(generated_notes) % 2 == 0 else 0.5)
        else:
            # Create a new prediction array with only valid notes
            adjusted_prediction = np.zeros_like(prediction)
            for idx in valid_indices:
                adjusted_prediction[idx] = prediction[idx]
            adjusted_prediction = adjusted_prediction / np.sum(adjusted_prediction)  # Re-normalize

            # Select the next note based on adjusted AI probabilities
            index = np.random.choice(len(adjusted_prediction), p=adjusted_prediction)
            predicted_pattern = int_to_note[index]

            # Append the predicted note and duration
            generated_notes.append(predicted_pattern)
            weighted_durations = np.random.choice(
                durations['values'], p=durations['weights']
            )
            generated_durations.append(weighted_durations)

            print(f"AI selected valid note: {predicted_pattern} with adjusted probability.")

            # Prepare the next input sequence
            next_features = [index] * 5  # Replace with appropriate features if necessary
            next_features_array = np.array([next_features]).reshape((1, 1, 5))
            current_sequence = np.append(current_sequence, next_features_array, axis=1)
            current_sequence = current_sequence[:, 1:]  # Keep sequence length constant

    return generated_notes, generated_durations


def create_midi(generated_notes, generated_durations, output_file, tempo_bpm=120):
    output_stream = stream.Stream()
    # Add tempo to the stream
    output_tempo = tempo.MetronomeMark(number=tempo_bpm)
    output_stream.append(output_tempo)

    print(generated_notes)
    print(generated_durations)

    for i, note_name in enumerate(generated_notes):
        duration = generated_durations[i]

        # Create a new note object and assign its duration
        new_note = note.Note(note_name)
        new_note.duration.quarterLength = duration  # Set the note's duration
        new_note.storedInstrument = instrument.Piano()  # Ensure it's for piano

        output_stream.append(new_note)  # Add the note to the stream

    # Save the Stream to a MIDI file
    output_stream.write('midi', fp=output_file)


# Main method that calls all the other ones, and the one that'll be used later
def main_lstm_generate_music(
        amount_of_notes, valid_notes, range_lower, range_upper,
        tempo, temperature, durations, output_path
):
    model = load_LSTM_model(sequence_length=100, n_unique_notes=576,
                            checkpoint_path=r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend'
                                            r'\generation\trained-models\weights-epoch-30-loss-1.7863-acc-0.5304'
                                            r'.weights.h5')

    # Load mappings
    note_to_int, int_to_note, rest_indices = load_mappings()

    # Prepare the start sequence
    start_sequence = load_start_sequence()

    durations.reverse()

    # Generate music
    valid_scale_notes = valid_notes  # Pass the array from frontend
    durations = {'values': [0.25, 0.5, 1, 2, 4], 'weights': durations}  # Weights from frontend
    notes, durations = generate_music_lstm(
        model, start_sequence, int_to_note, rest_indices, valid_scale_notes, durations,
        n_generate=amount_of_notes, temperature=temperature, lower_range=range_lower, upper_range=range_upper
    )

    # Create MIDI
    create_midi(notes, durations, output_file=output_path, tempo_bpm=tempo)

    print("Music generation complete.")

    return output_path


# # Test main method
# main_lstm_generate_music('LSTM', 100, ['C#', 'D', 'E', 'F#', 'G#', 'A', 'B'], 40, 45, 100, 0.5,
#                          [0, 0.3, 0.2, 0.15, 0.35])
