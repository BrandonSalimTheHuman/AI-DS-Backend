import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from music21 import stream, note, instrument, tempo
import random


# Method to build the gru model
def build_gru_model(sequence_length, n_unique_notes):
    model = Sequential()
    model.add(GRU(512, input_shape=(sequence_length, 1), return_sequences=True))  # Adjust input shape to match GRU
    model.add(Dropout(0.3))
    model.add(GRU(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_unique_notes, activation='softmax'))
    return model


# Method that calls the other method that builds the model, then loads the trained weights
def load_gru_model(sequence_length, n_unique_notes, checkpoint_path):
    model = build_gru_model(sequence_length, n_unique_notes)
    model.load_weights(checkpoint_path)

    return model


# Method to load mappings
def load_mappings(dataset_path):
    try:
        mapping_path = os.path.join(dataset_path, "note_to_int.pkl")
        print(f"Loading mapping from: {mapping_path}")
        if not os.path.exists(mapping_path):
            raise Exception(f"Model file not found at: {mapping_path}")
        with open(mapping_path, 'rb') as f:
            note_to_int = pickle.load(f)
        int_to_note = {number: note for note, number in note_to_int.items()}
        rest_indices = [index for index, note in int_to_note.items() if note.lower() == 'rest' or note == 'R']
        return note_to_int, int_to_note, rest_indices
    except Exception as e:
        print(f"Error loading mappings: {e}")
        raise


# Method to choose star sequence
def load_start_sequence(dataset_path):
    try:
        training_data_path = os.path.join(dataset_path, "input_sequences_notes.pkl")
        print(f"Loading training data from: {training_data_path}")
        if not os.path.exists(training_data_path):
            raise Exception(f"Model file not found at: {training_data_path}")
        with open(training_data_path, 'rb') as f:
            X_notes = pickle.load(f)
        start_sequence = X_notes[np.random.randint(0, len(X_notes))]
        return start_sequence
    except Exception as e:
        print(f"Error loading sequence: {e}")
        raise


# Helper function to convert note to MIDI number representation
def note_to_midi(note):
    note_mapping = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
        "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
    }

    # Separate note and octave
    note_name = note[:-1]
    octave = int(note[-1])

    # Calculate MIDI number
    midi_number = 12 * (octave + 1) + note_mapping[note_name]
    return midi_number


# Main method to generate the music using GRU
def generate_music_gru(model, start_sequence, int_to_note, rest_indices, valid_scale_notes, durations, n_generate=100,
                       temperature=1.0, lower_range=60, upper_range=70):
    generated_notes = []
    generated_durations = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), 1))  # Reshape to match GRU input

    # Define C Major scale and octave of C4
    valid_octave = 4  # C4 to B4

    while len(generated_notes) < n_generate:
        # Predict the next note
        prediction = model.predict(current_sequence, verbose=0)[0]

        # Zero out probabilities for rests
        prediction[rest_indices] = 0

        # Apply temperature scaling
        prediction = prediction ** (1 / temperature)
        prediction = prediction / np.sum(prediction)  # Renormalize

        # Select the next note
        index = np.random.choice(len(prediction), p=prediction)
        predicted_note = int_to_note[index]

        # Skip chords (notes containing a dot, e.g., '11.2') and invalid patterns
        if '.' in predicted_note or predicted_note.isdigit():
            print(f"Skipping chord or invalid note: {predicted_note}")
            continue

        # Check if the predicted note is valid
        if predicted_note[:-1] in valid_scale_notes and lower_range <= note_to_midi(predicted_note) <= upper_range:
            print(f"Note kept: {predicted_note}")
            snapped_note = predicted_note  # Keep the note as-is
        else:
            print(f"Note {predicted_note} is outside constraints. Snapping...")

            # Filter probabilities to keep only valid scale notes in the C4 octave
            scale_filtered_prediction = np.zeros_like(prediction)
            for idx, note_name in int_to_note.items():
                if note_name[:-1] in valid_scale_notes and lower_range <= note_to_midi(note_name) <= upper_range:
                    scale_filtered_prediction[idx] = prediction[idx]

            # If no valid notes remain, fall back to random snapping
            if np.sum(scale_filtered_prediction) == 0:
                snapped_note = random.choice(valid_scale_notes) + str(valid_octave)
                print(f"Snapped note NOT using AI probabilities")
            else:
                # Re-normalize probabilities
                scale_filtered_prediction = scale_filtered_prediction / np.sum(scale_filtered_prediction)

                # Select the snapped note based on AI probabilities
                index = np.random.choice(len(scale_filtered_prediction), p=scale_filtered_prediction)
                snapped_note = int_to_note[index]
                # print(scale_filtered_prediction)
                print(f"Snapped note using AI probabilities")

            print(f"Snapped note: {predicted_note} -> {snapped_note}")

        # Append the snapped note and a fixed duration
        generated_notes.append(snapped_note)

        weighted_durations = np.random.choice(
            durations['values'], p=durations['weights']
        )
        generated_durations.append(weighted_durations)

        # Print the generated note and its duration
        print(f"Generated note: {snapped_note}, Duration: {generated_durations[-1]}")

        # Update the current sequence with the new prediction
        next_features = np.array([[[index]]])  # Ensure 3D shape (1, 1, 1)
        current_sequence = np.append(current_sequence, next_features, axis=1)
        current_sequence = current_sequence[:, 1:]  # Keep sequence length constant

    return generated_notes, generated_durations


# Coverting notes to MIDI
def create_midi(generated_notes, generated_durations, output_file, tempo_bpm=120):
    output_stream = stream.Stream()

    # Add tempo to the stream
    output_tempo = tempo.MetronomeMark(number=tempo_bpm)
    output_stream.append(output_tempo)

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
def main_gru_generate_music(
        amount_of_notes, valid_notes, range_lower, range_upper,
        tempo, temperature, durations, dataset_path, model_weights_path, output_path
):
    try:
        model_path = os.path.join(model_weights_path, "weights-epoch-15-loss-3.9767-acc-0.2276.weights.h5")
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found at: {model_path}")
        model = load_gru_model(sequence_length=50, n_unique_notes=576,
                               checkpoint_path=model_path)
    except Exception as e:
        print(f"Error loading mappings: {e}")
        raise

    # Load mappings
    note_to_int, int_to_note, rest_indices = load_mappings(dataset_path)

    # Prepare the start sequence
    start_sequence = load_start_sequence(dataset_path)

    durations.reverse()

    # Generate music
    valid_scale_notes = valid_notes  # Pass the array from frontend
    durations = {'values': [0.25, 0.5, 1, 2, 4], 'weights': durations}  # Weights from frontend
    notes, durations = generate_music_gru(
        model, start_sequence, int_to_note, rest_indices, valid_scale_notes, durations,
        n_generate=amount_of_notes, temperature=temperature, lower_range=range_lower, upper_range=range_upper
    )

    # Create MIDI
    create_midi(notes, durations, output_file=output_path, tempo_bpm=tempo)

    print("Music generation complete.")

    return output_path
