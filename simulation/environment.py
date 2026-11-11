"""Runnable negotiation loop: seller actions, buyer counteroffers, terminal reward."""
from data.preprocess import PRICE_TOKEN
from models.generator import render
from simulation.buyers import make_buyer
from config import REJECTION_REWARD, SUCCESS_MULTIPLIER, N_REPLIES, N_PRICES


class RuleSeller:
    """Rule-based seller benchmark (explicitly not from the CHAI paper)."""
    def decide(self, state, **_):
        buyer_offer = next((h['price'] for h in reversed(state['history'])
                            if h['role'] == 'buyer' and h['kind'] == 'offer'), None)
        if buyer_offer is not None and buyer_offer >= .82:
            return {'kind': 'accept', 'text': 'accept', 'price': buyer_offer}, []
        price = max(.78, min(.95, state['last_price'] * .96))
        return {'kind': 'offer', 'text': 'offer', 'price': price}, []


class LMSeller:
    def __init__(self, generator):
        self.generator = generator

    def decide(self, state, n_replies=N_REPLIES, n_prices=N_PRICES, **_):
        from rl.prop import candidates
        import random
        possibilities = candidates(state, self.generator, n_replies, n_prices)
        return random.choice(possibilities), []


def simulate(seller, style, title='Used wooden desk', list_price=100., max_turns=6,
             reservation=None, verbose=False, show_q=False,
             n_replies=N_REPLIES, n_prices=N_PRICES):
    buyer = make_buyer(style, reservation)
    history = [{'role': 'buyer', 'kind': 'offer', 'text': 'offer', 'price': buyer.first_offer()}]
    current_price = buyer.first_offer()
    transcript = [('buyer', f'I can pay ${current_price*list_price:.2f}.')]
    deal = False
    final_price = None
    fallback_seller_turns = 0
    for turn in range(1, max_turns + 1):
        state = {'title': title, 'description': 'Used item in good condition.',
                 'list_price': float(list_price), 'history': list(history),
                 'last_price': current_price, 'last_type': history[-1]['kind']}
        action, _ = seller.decide(state, n_replies=n_replies, n_prices=n_prices,
                                  show=bool(verbose and show_q and turn == 1))
        if action.get('source') == 'gpt2_fallback':
            fallback_seller_turns += 1
            if verbose:
                print('WARNING: this seller turn uses a flagged GPT-2 emergency fallback.')
        utterance = render(action, list_price)
        transcript.append(('seller', utterance))
        if action['kind'] == 'accept':
            # Cannot accept without a currently outstanding buyer offer.
            if history[-1]['role'] == 'buyer' and history[-1]['kind'] == 'offer':
                deal, final_price = True, current_price * list_price
                break
            # Invalid acceptance counts as nonterminal chatter.
        elif action['kind'] == 'reject':
            break
        if action['kind'] == 'offer' or (action['kind'] == 'message' and PRICE_TOKEN in action['text']):
            proposed = float(action.get('price') or current_price)
            reply, bid = buyer.reply(proposed, turn)
            history.append({'role': 'seller', **action})
            history.append({'role': 'buyer', 'kind': reply, 'text': reply, 'price': bid})
            if reply == 'accept':
                deal, final_price = True, proposed * list_price
                transcript.append(('buyer', 'I accept that offer.'))
                break
            if reply == 'reject':
                transcript.append(('buyer', 'Sorry, I cannot pay that much.'))
                break
            current_price = bid
            transcript.append(('buyer', f'Could you do ${bid*list_price:.2f}?'))
        else:
            history.append({'role': 'seller', **action})
            # A buyer repeats the current bid if the seller didn't make an offer.
            history.append({'role': 'buyer', 'kind': 'offer', 'text': 'offer', 'price': current_price})
            transcript.append(('buyer', f'My offer is ${current_price*list_price:.2f}.'))
    reward = SUCCESS_MULTIPLIER * final_price / list_price if deal else REJECTION_REWARD
    if verbose:
        for speaker, text in transcript:
            print(f'{speaker.upper():>6}: {text}')
        print(f'Agreement={deal}, final_price={final_price}, reward={reward:.2f}, turns={len([x for x in transcript if x[0]=="seller"])}')
    return {'deal': deal, 'final_price': final_price, 'reward': reward,
            'turns': sum(x[0] == 'seller' for x in transcript), 'transcript': transcript,
            'style': style, 'fallback_seller_turns': fallback_seller_turns}
