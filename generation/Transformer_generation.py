import os
import pickle
import numpy as np
from music21 import stream, note, instrument, tempo
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
)
import random


# Define the Transformer Encoder Block
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def positional_encoding(self, sequence_length, d_model):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = np.zeros((sequence_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]
        return tf.cast(pe, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = x + res
    return x


# # Build the model
# sequence_length = 100
# feature_dim = 5
# n_notes = 576

def build_transformer_model(sequence_length, feature_dim, n_notes):
    embedding_dim = 512
    num_transformer_blocks = 2
    head_size = 64
    num_heads = 8
    ff_dim = 256
    dropout = 0.3

    inputs = Input(shape=(sequence_length, feature_dim))
    x = Dense(embedding_dim)(inputs)
    x = PositionalEncoding(sequence_length, embedding_dim)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)

    outputs = Dense(n_notes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


# Load the model and weights
# model = build_transformer_model(sequence_length, feature_dim, n_notes)
# weights_path = r'trained-models\weights-epoch-10-loss-4.1157-acc-0.2215.weights.h5'
# model.load_weights(weights_path)

# Method that calls the other method that builds the model, then loads the trained weights
def load_transformer_model(sequence_length, n_unique_notes, feature_dim,
                           checkpoint_path=r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend'
                                           r'\generation\trained-models\weights-epoch-10-loss-4.1157-acc-0.2215'
                                           r'.weights.h5'):
    model = build_transformer_model(sequence_length, feature_dim, n_unique_notes)
    model.load_weights(checkpoint_path)

    return model


# Method to load mappings
def load_mappings(mapping_path=r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject-Backend\generation'
                               r'\Subset_Dataset\note_to_int.pkl'):
    with open(mapping_path, 'rb') as f:
        note_to_int = pickle.load(f)
    int_to_note = {number: note for note, number in note_to_int.items()}
    rest_indices = [index for index, note in int_to_note.items() if note.lower() == 'rest' or note == 'R']
    return note_to_int, int_to_note, rest_indices


# Method to choose star sequence
def load_start_sequence(sequence_length, feature_dim):
    start_sequence = np.random.rand(sequence_length, feature_dim)
    return start_sequence


# # Load the mappings and data
# with open('Subset_Dataset/note_to_int.pkl', 'rb') as f:
#     note_to_int = pickle.load(f)
#
# int_to_note = {number: note for note, number in note_to_int.items()}
#
# # Define rest indices
# rest_indices = [index for index, note_str in int_to_note.items() if note_str.lower() == 'rest' or note_str == 'R']

# Define valid C Major notes and MIDI range
# c_major_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
# valid_midi_range = range(60, 71)  # MIDI values for C4 to B5 inclusive

# Define function to generate music
def generate_music_transformer(
        model,
        start_sequence,
        int_to_note,
        rest_indices,
        valid_scale_notes,
        durations,
        feature_dim,
        n_generate=100,
        temperature=1.0,
        lower_range=60,
        upper_range=70
):
    generated_notes = []
    generated_durations = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), feature_dim))
    # Range excludes the upper limit, so add 1 to it
    valid_midi_range = range(lower_range, upper_range + 1)

    while len(generated_notes) < n_generate:
        predictions = model.predict(current_sequence, verbose=0)[0]

        # Zero out probabilities for rests
        predictions[rest_indices] = 0

        # Apply temperature scaling
        predictions = predictions ** (1 / temperature)
        predictions = predictions / np.sum(predictions)  # Renormalize

        # Filter the predictions to only include notes in C major scale
        scale_filtered_prediction = np.zeros_like(predictions)
        for idx, note_str in int_to_note.items():
            note_name = note_str[:-1]  # Remove octave number
            if note_name in valid_scale_notes:
                scale_filtered_prediction[idx] = predictions[idx]

        # If no valid notes in the scale, pick a valid note based on AI probabilities
        if np.sum(scale_filtered_prediction) == 0:
            valid_indices = [
                i for i, note_str in int_to_note.items()
                if note_str[:-1] in valid_scale_notes and
                   note.Note(note_str).pitch.midi in valid_midi_range
            ]
            if valid_indices:
                valid_probabilities = [predictions[i] for i in valid_indices]
                valid_probabilities = valid_probabilities / np.sum(valid_probabilities)
                index = np.random.choice(valid_indices, p=valid_probabilities)
                snapped_note = int_to_note[index]
                print(f"Snapped to valid note based on AI probabilities: {snapped_note} (original: out of scale)")
                predicted_note = snapped_note
            else:
                raise ValueError("No valid notes found in int_to_note.")
        else:
            scale_filtered_prediction = scale_filtered_prediction / np.sum(scale_filtered_prediction)
            index = np.random.choice(len(scale_filtered_prediction), p=scale_filtered_prediction)
            predicted_note = int_to_note[index]
            print(f"AI generated note: {predicted_note} with probability: {scale_filtered_prediction[index]}")

        # Ensure only single notes (no chords)
        if ('.' in predicted_note) or predicted_note.isdigit():
            print(f"Skipping chord or invalid pattern: {predicted_note}")
            continue  # Skip chord-like patterns

        # Replace G4 with another random note from the list
        # THIS IS JUST A TEST. IT JUST MAKES ANOTHER NOTE GET SPAMMED.
        # if predicted_note == 'G4':
        #     print(f"G4 detected. Replacing with a random note from C Major based on weights.")
        #     valid_indices = [
        #         i for i, note_str in int_to_note.items()
        #         if note_str[:-1] in c_major_notes and
        #         note.Note(note_str).pitch.midi in valid_midi_range and
        #         note_str != 'G4'  # Exclude G4 from valid replacements
        #     ]
        #     if valid_indices:
        #         valid_probabilities = [predictions[i] for i in valid_indices]
        #         valid_probabilities = valid_probabilities / np.sum(valid_probabilities)
        #         index = np.random.choice(valid_indices, p=valid_probabilities)
        #         predicted_note = int_to_note[index]
        #         print(f"Replaced G4 with: {predicted_note}")
        #     else:
        #         raise ValueError("No valid notes in C Major for G4 replacement found.")

        # Ensure note is within valid MIDI range
        note_midi = note.Note(predicted_note).pitch.midi
        if note_midi not in valid_midi_range:
            # Snap to valid note based on AI probabilities
            valid_indices = [
                i for i, note_str in int_to_note.items()
                if note_str[:-1] in valid_scale_notes and
                   note.Note(note_str).pitch.midi in valid_midi_range
            ]
            if valid_indices:
                valid_probabilities = [predictions[i] for i in valid_indices]
                valid_probabilities = valid_probabilities / np.sum(valid_probabilities)
                index = np.random.choice(valid_indices, p=valid_probabilities)
                snapped_note = int_to_note[index]
                print(f"Snapped to valid range: {snapped_note} (original: {predicted_note})")
                predicted_note = snapped_note
            else:
                raise ValueError("No valid notes within MIDI range found.")

        # Append the predicted note
        generated_notes.append(predicted_note)

        weighted_durations = np.random.choice(
            durations['values'], p=durations['weights']
        )
        generated_durations.append(weighted_durations)

        print(
            f"Generated {len(generated_notes)}/{n_generate} notes: {predicted_note} with duration {weighted_durations}")

        # Prepare the next input sequence
        next_features = np.array([[index] * feature_dim]).reshape((1, 1, feature_dim))
        current_sequence = np.append(current_sequence, next_features, axis=1)
        current_sequence = current_sequence[:, 1:]  # Keep sequence length constant

    return generated_notes, generated_durations


# Create MIDI file from generated notes and durations
def create_midi(
        generated_notes,
        generated_durations,
        output_file,
        tempo_bpm=120
):
    output_stream = stream.Stream()
    output_tempo = tempo.MetronomeMark(number=tempo_bpm)
    output_stream.append(output_tempo)

    for note_name, duration in zip(generated_notes, generated_durations):
        try:
            new_note = note.Note(note_name)
            new_note.duration.quarterLength = duration
            new_note.storedInstrument = instrument.Piano()
            output_stream.append(new_note)
        except Exception as e:
            print(f"Error creating note {note_name}: {e}")

    output_stream.write('midi', fp=output_file)


# # Generate music
# start_sequence = np.random.rand(sequence_length, feature_dim)  # Replace with your actual start sequence
# generated_notes, generated_durations = generate_music_transformer(
#     model,
#     start_sequence,
#     int_to_note,
#     rest_indices,
#     n_generate=75,
#     temperature=8
# )
#
# # Save to MIDI
# create_midi(
#     generated_notes,
#     generated_durations,
#     output_file='generated_music_result/A_generated_music_transformer.mid'
# )
#
# print("Music generation complete. Saved to 'A_generated_music_transformer.mid'.")


# Main method that calls all the other ones, and the one that'll be used later
def main_transformer_generate_music(
        amount_of_notes, valid_notes, range_lower, range_upper,
        tempo, temperature, durations, output_path
):
    model = load_transformer_model(sequence_length=100, n_unique_notes=576, feature_dim=5,
                                   checkpoint_path=r'C:\Users\Brandon Salim\PycharmProjects\TASem3\AI-FinalProject'
                                                   r'-Backend\generation\trained-models\weights-epoch-10-loss-4.1157'
                                                   r'-acc-0.2215.weights.h5')

    # Load mappings
    note_to_int, int_to_note, rest_indices = load_mappings()

    # Prepare the start sequence
    start_sequence = load_start_sequence(feature_dim=5, sequence_length=100)

    durations.reverse()

    # Generate music
    valid_scale_notes = valid_notes  # Pass the array from frontend
    durations = {'values': [0.25, 0.5, 1, 2, 4], 'weights': durations}  # Weights from frontend
    notes, durations = generate_music_transformer(
        model, start_sequence, int_to_note, rest_indices, valid_scale_notes, durations,
        feature_dim=5, n_generate=amount_of_notes, temperature=temperature, lower_range=range_lower,
        upper_range=range_upper
    )

    # Create MIDI
    create_midi(notes, durations, output_file=output_path, tempo_bpm=tempo)

    print("Music generation complete. Saved to 'generated_music_transformer_E10.mid'.")

    return output_path


# # Test main method
# main_transformer_generate_music('Transformer', 100, ['C', 'D', 'E', 'F', 'G', 'A', 'B'], 40, 45, 100, 0.5,
#                                 [0, 0.3, 0.2, 0.15, 0.35])
