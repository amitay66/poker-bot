import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from treys import Evaluator, Card
import random

class PokerBot:
    def __init__(self):
        self.model = self.create_random_forest_model()
        self.scaler = MinMaxScaler()
        self.training_data = []
        self.evaluator = Evaluator()
        self.model_trained = False

    def create_random_forest_model(self):
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def extract_features(self, hand, position, pot_size, board_cards, stack_size, num_players, call_amount=50, future_bets=100):
        features = [
            int(hand[0][0] in ['A', 'K']),
            int(hand[1][0] in ['A', 'K']),
            len(board_cards),
            self.position_to_index(position),
            pot_size / 1000,
            1.0 / (self.calculate_hand_strength(hand, board_cards) + 1),
            self.evaluate_board_strength(board_cards),
            self.calculate_opponent_aggressiveness(),
            self.calculate_opponent_hand_strength(hand, board_cards),
            self.calculate_potential_winning_chance(hand, board_cards),
            self.calculate_pot_odds(pot_size, call_amount),
            self.calculate_implied_odds(pot_size, call_amount, future_bets),
            stack_size / 1000,
            num_players
        ]
        return features + [0] * (15 - len(features))

    def position_to_index(self, position):
        positions = ['early', 'middle', 'late']
        return positions.index(position) if position in positions else -1

    def scale_features(self, features):
        if not hasattr(self.scaler, 'mean_'):
            return features
        return self.scaler.transform([features])[0]

    def train_model(self):
        if len(self.training_data) < 100:
            print("Not enough data to train the model.")
            return

        df = pd.DataFrame(self.training_data)
        X = df.drop(columns='action').values
        y = df['action'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")

        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        self.model_trained = True

    def calculate_hand_strength(self, hand, board_cards):
        try:
            hand_cards = [Card.new(card) for card in hand]
            board_cards = [Card.new(card) for card in board_cards]
            hand_strength = self.evaluator.evaluate(board_cards, hand_cards)
            max_strength = 7462
            normalized_strength = (max_strength - hand_strength) / max_strength
            return normalized_strength
        except Exception as e:
            print(f"Error in calculating hand strength: {e}")
            return 0

    def calculate_opponent_hand_strength(self, hand, board_cards):
        return self.calculate_hand_strength(hand, board_cards)

    def evaluate_board_strength(self, board_cards):
        if not board_cards:
            return 0
        return np.random.random()

    def calculate_opponent_aggressiveness(self):
        return np.random.random()

    def calculate_potential_winning_chance(self, hand, board_cards):
        return np.random.random()

    def calculate_pot_odds(self, pot_size, call_amount):
        if call_amount == 0:
            return 0
        total_pot = pot_size + call_amount
        return total_pot / call_amount

    def calculate_implied_odds(self, pot_size, call_amount, future_bets):
        if call_amount == 0:
            return 0
        implied_pot = pot_size + future_bets
        return implied_pot / call_amount

    def estimate_opponent_range(self, opponent_behavior):
        if opponent_behavior == 'aggressive':
            return [['As', 'Ks'], ['Ad', 'Kd'], ['Qh', 'Qs']]
        elif opponent_behavior == 'conservative':
            return [['Ac', 'Ah'], ['Kc', 'Kh'], ['Qc', 'Qs']]
        else:
            return [['2c', '2h'], ['3d', '3s']]

    def decide_action(self, hand, position, pot_size, board_cards, stack_size, num_players):
        if not self.model_trained:
            print("Model not trained yet.")
            return 'fold'

        features = self.extract_features(hand, position, pot_size, board_cards, stack_size, num_players)
        features_scaled = self.scale_features(features)
        prediction = self.model.predict([features_scaled])
        actions = ['fold', 'call', 'raise']
        return actions[np.argmax(prediction)]

    def add_training_data(self, hand, position, pot_size, board_cards, stack_size, num_players, action):
        features = self.extract_features(hand, position, pot_size, board_cards, stack_size, num_players)
        self.training_data.append({
            'feature1': features[0], 'feature2': features[1], 'feature3': features[2],
            'feature4': features[3], 'feature5': features[4], 'feature6': features[5],
            'feature7': features[6], 'feature8': features[7], 'feature9': features[8],
            'feature10': features[9], 'feature11': features[10], 'feature12': features[11],
            'feature13': features[12], 'feature14': features[13], 'feature15': features[14],
            'action': action
        })
