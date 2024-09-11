from poker_bot import PokerBot
from utils import generate_random_training_data

def main():
    bot = PokerBot()

    # Add training data
    training_data = generate_random_training_data(120)
    for example in training_data:
        bot.add_training_data(
            hand=example['hand'],
            position=example['position'],
            pot_size=example['pot_size'],
            board_cards=example['board_cards'],
            stack_size=example['stack_size'],
            num_players=example['num_players'],
            action=example['action']
        )

    # Train the bot
    bot.train_model()

    # Check the bot
    hand = ['As', 'Ks']
    position = 'middle'
    pot_size = 150
    board_cards = ['2d', '3h', '5s']
    stack_size = 1000
    num_players = 3
    action = bot.decide_action(hand, position, pot_size, board_cards, stack_size, num_players)
    print(f"Bot's action: {action}")

if __name__ == "__main__":
    main()
