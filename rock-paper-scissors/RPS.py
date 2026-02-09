def player(prev_play):
    """
    prev_play: opponent's last move ('' on first call of each match)
    Returns: 'R', 'P', or 'S'
    State is stored on player.state (initialized on first call).
    """

    moves = ['R','P','S']
    beats = {'R':'P','P':'S','S':'R'}   # beats[x] beats x
    # inverse mapping: move that x beats
    loses_to = {v:k for k,v in beats.items()}

    # Initialize persistent state container (survives across calls)
    if not hasattr(player, 'state'):
        player.state = {
            # per-match histories (reset at start of each match)
            'opp_hist': [],
            'my_hist': [],
            'round_idx': 0,
            # predictor bookkeeping
            'predictor_scores': {
                'repeat_last': 1.0, 'bigram': 1.0, 'last2': 1.0,
                'freq': 1.0, 'cycle': 1.0, 'copy_my': 1.0,
                'react_to_my_last': 1.0, 'anti_myfreq': 1.0
            },
            'last_predictions': {},
            # abbey detector state
            'abbey_state': {'active': False, 'timer': 0, 'detect_weight': 0.0}
        }

    S = player.state
    opp_hist = S['opp_hist']
    my_hist = S['my_hist']
    predictor_scores = S['predictor_scores']
    last_predictions = S['last_predictions']
    abbey_state = S['abbey_state']

    # Match-start reset: harness calls player('') at match beginning
    if prev_play == '':
        # Clear match-local state
        opp_hist.clear()
        my_hist.clear()
        S['round_idx'] = 0
        last_predictions.clear()
        # reset predictor scores to neutral
        for k in list(predictor_scores.keys()):
            predictor_scores[k] = 1.0
        # reset abbey detector
        abbey_state.clear()
        abbey_state.update({'active': False, 'timer': 0, 'detect_weight': 0.0})
        # First move seed
        S['round_idx'] += 1
        my_move = 'R'
        my_hist.append(my_move)
        return my_move

    # Only valid moves are 'R', 'P', 'S'
    if prev_play in moves:
        # Update predictor scoring from last round
        if last_predictions:
            # decay old scores
            decay = 0.92
            for k in predictor_scores:
                predictor_scores[k] *= decay
            # reward accurate predictors
            for name, pred in last_predictions.items():
                if pred == prev_play:
                    predictor_scores[name] = predictor_scores.get(name, 0.0) + 1.4

        # record opponent move
        opp_hist.append(prev_play)
        S['round_idx'] += 1

    # Convenience copies and lengths
    opp = [m for m in opp_hist if m in moves]
    me = [m for m in my_hist if m in moves]
    n = len(opp)

    # Helper: most frequent element
    def most_frequent(seq):
        if not seq:
            return None
        cnt = {}
        for x in seq:
            cnt[x] = cnt.get(x, 0) + 1
        return max(cnt.items(), key=lambda t: (t[1], t[0]))[0]

    # First-turn safety: if we somehow have no history, play R
    if n == 0 and len(me) == 0:
        my_move = 'R'
        my_hist.append(my_move)
        last_predictions.clear()
        return my_move

    # Strict Abbey detection
    my_most = most_frequent(me)
    abbey_expected = beats[my_most] if my_most else None
    abbey_detect_rate = 0.0
    if abbey_expected and len(opp) > 0:
        abbey_detect_rate = sum(1 for x in opp if x == abbey_expected) / len(opp)

    # Exponential moving average for detection
    alpha = 0.20
    abbey_state['detect_weight'] = (1 - alpha) * abbey_state.get('detect_weight', 0.0) + alpha * abbey_detect_rate

    # Enter abbey mode only on strong, sustained evidence
    # require minimum personal history so my_most is reliable
    if not abbey_state.get('active', False):
        if len(me) >= 12 and abbey_state['detect_weight'] > 0.50:
            abbey_state['active'] = True
            abbey_state['timer'] = 24
    else:
        # while active, refresh if evidence remains
        if abbey_state['detect_weight'] > 0.35:
            abbey_state['timer'] = max(abbey_state['timer'], 12)
        abbey_state['timer'] -= 1
        if abbey_state['timer'] <= 0:
            abbey_state['active'] = False
            abbey_state['timer'] = 0
            abbey_state['detect_weight'] *= 0.4

    # If in abbey mode, force self exploitation (play my_most)
    if abbey_state.get('active', False) and my_most:
        my_hist.append(my_most)
        # we don't alter last_predictions here but keep learning
        return my_most

    # Build predictors
    preds = {}
    confs = {}

    # repeat_last
    if n >= 1:
        preds['repeat_last'] = opp[-1]
        confs['repeat_last'] = 0.35 if n < 4 else 0.6

    # bigram
    if n >= 2:
        trans = {}
        for i in range(n-1):
            a,b = opp[i], opp[i+1]
            trans.setdefault(a, {})
            trans[a][b] = trans[a].get(b, 0) + 1
        last = opp[-1]
        if last in trans:
            succ = max(trans[last].items(), key=lambda x:(x[1], x[0]))[0]
            preds['bigram'] = succ
            total = sum(trans[last].values())
            confs['bigram'] = trans[last][succ] / total if total>0 else 0.0

    # last 2 mapping
    if n >= 3:
        mapping = {}
        for i in range(n-2):
            key = opp[i] + opp[i+1]
            mapping.setdefault(key, {})
            nxt = opp[i+2]
            mapping[key][nxt] = mapping[key].get(nxt, 0) + 1
        key = opp[-2] + opp[-1]
        if key in mapping:
            next_move = max(mapping[key].items(), key=lambda x:(x[1], x[0]))[0]
            preds['last2'] = next_move
            total = sum(mapping[key].values())
            confs['last2'] = mapping[key][next_move] / total if total>0 else 0.0

    # freq (recent window)
    if n >= 1:
        window = opp[-30:] if n>30 else opp
        freq = {}
        for m in window:
            freq[m] = freq.get(m, 0) + 1
        most = max(freq.items(), key=lambda x:(x[1], x[0]))[0]
        preds['freq'] = most
        confs['freq'] = freq[most] / len(window)

    # cycle detection
    def detect_cycle(seq, maxL=8):
        Lmax = min(maxL, max(1, len(seq)))
        for L in range(1, Lmax+1):
            if len(seq) >= 2*L and seq[-L:] == seq[-2*L:-L]:
                return seq[-L:], L
        for L in range(1, Lmax+1):
            pattern = seq[-L:]
            if ''.join(pattern) in ''.join(seq[:-L]):
                return pattern, L
        return None, None

    pattern, L = detect_cycle(opp, maxL=8)
    if pattern:
        next_index = len(opp) % L
        preds['cycle'] = pattern[next_index]
        confs['cycle'] = 0.92 if len(opp) >= 2*L else 0.6

    # copy_my predictor
    if len(me) >= 2 and n >= 2:
        copies = 0
        total_pairs = 0
        limit = min(len(me), len(opp)-1)
        for i in range(limit):
            total_pairs += 1
            if opp[i+1] == me[i]:
                copies += 1
        if total_pairs > 0:
            rate = copies / total_pairs
            if rate > 0.46:
                preds['copy_my'] = me[-1]
                confs['copy_my'] = rate

    # react_to_my_last predictor
    if len(me) >= 1 and n >= 1:
        last_my = me[-1]
        expected = beats[last_my]
        count_expected = sum(1 for x in opp if x == expected)
        rate = count_expected / len(opp) if len(opp)>0 else 0.0
        if rate > 0.32:
            preds['react_to_my_last'] = expected
            confs['react_to_my_last'] = min(0.92, rate + 0.05)

    # anti_myfreq: opponent often counters my most frequent -> predict that counter
    if len(me) >= 8:
        window_me = me[-40:] if len(me)>40 else me
        freq_me = {}
        for m in window_me:
            freq_me[m] = freq_me.get(m, 0) + 1
        my_most2 = max(freq_me.items(), key=lambda x:(x[1], x[0]))[0]
        predicted_opp = beats[my_most2]
        count_pred = sum(1 for x in opp if x == predicted_opp)
        if len(opp)>0:
            rate2 = count_pred / len(opp)
            if rate2 > 0.30:
                preds['anti_myfreq'] = predicted_opp
                confs['anti_myfreq'] = min(0.9, rate2 + 0.08)

    # fallback
    if not preds:
        preds['repeat_last'] = opp[-1] if n>=1 else 'R'
        confs['repeat_last'] = 0.4

    # combine: historical strength * confidence
    combined = {}
    for name, pred in preds.items():
        conf = confs.get(name, 0.2)
        base = predictor_scores.get(name, 1.0)
        combined[name] = base * conf

    # choose best predictor deterministically
    best_name = max(sorted(combined.keys()), key=lambda k:(combined[k], k))
    predicted_move = preds[best_name]

    # store last predictions for next round credit
    last_predictions.clear()
    for name, pred in preds.items():
        last_predictions[name] = pred

    # choose move that beats predicted opponent move
    my_move = beats.get(predicted_move, 'R')
    my_hist.append(my_move)
    return my_move
