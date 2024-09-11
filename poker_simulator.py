import random
from poker_bot import PokerBot

class PokerSimulator:
    def __init__(self, num_players=3, starting_stack=1000):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.players = [PokerBot() for _ in range(num_players)]
        self.player_stacks = [starting_stack for _ in range(num_players)]
        self.pot = 0
        self.board_cards = []

    def deal_cards(self):
        deck = self.create_deck()
        hands = [[deck.pop(), deck.pop()] for _ in range(self.num_players)]
        return hands

    def create_deck(self):
        suits = ['s', 'h', 'd', 'c']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        deck = [f"{v}{s}" for v in values for s in suits]
        random.shuffle(deck)
        return deck

    def place_bets(self, min_bet=50):
        # Simulate each player placing a bet
        for i in range(self.num_players):
            bet = min_bet  # For now, a simple bet of 50 for all players
            if self.player_stacks[i] < bet:
                bet = self.player_stacks[i]  # If player doesn't have enough stack
            self.pot += bet
            self.player_stacks[i] -= bet
        print(f"Pot after bets: {self.pot}")

    def reveal_flop(self, deck):
        self.board_cards = [deck.pop(), deck.pop(), deck.pop()]
        print(f"Flop: {self.board_cards}")

    def reveal_turn(self, deck):
        self.board_cards.append(deck.pop())
        print(f"Turn: {self.board_cards[-1]}")

    def reveal_river(self, deck):
        self.board_cards.append(deck.pop())
        print(f"River: {self.board_cards[-1]}")

    def play_round(self):
        print("\nNew round starting...\n")

        # Deal cards to each player
        deck = self.create_deck()
        hands = self.deal_cards()
        print(f"Player hands: {hands}")

        # Players place bets
        self.place_bets()

        # Flop
        self.reveal_flop(deck)
        self.place_bets()

        # Turn
        self.reveal_turn(deck)
        self.place_bets()

        # River
        self.reveal_river(deck)
        self.place_bets()

        # Evaluate hands
        best_hand = -1
        winning_player = -1
        for i, player in enumerate(self.players):
            hand_strength = player.calculate_hand_strength(hands[i], self.board_cards)
            print(f"Player {i+1}'s hand strength: {hand_strength}")
            if hand_strength > best_hand:
                best_hand = hand_strength
                winning_player = i

        # Award the pot to the winning player
        print(f"Player {winning_player+1} wins the pot of {self.pot}!")
        self.player_stacks[winning_player] += self.pot
        self.pot = 0

        # Print out player stacks
        print("Player stacks after the round:")
        for i, stack in enumerate(self.player_stacks):
            print(f"Player {i+1}: {stack}")

    def run_simulation(self, num_rounds=10):
        for _ in range(num_rounds):
            self.play_round()


if __name__ == "__main__":
    sim = PokerSimulator(num_players=3, starting_stack=1000)
    sim.run_simulation(num_rounds=5)
