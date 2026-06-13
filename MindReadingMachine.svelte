<script lang="ts">

	type Record = [number, boolean];
	type Situation = [Record, Record];
	type Triple = [Record, Record, Record];

	let chosen = -1;
	let computerChoice = -1;
	let computerWinCount = 0;
	let playerWinCount = 0;

	// Initialize history with each of the 16 possible scenarios
	const history: Record[] = [];

	function choose(face: number) {
		// 0 is heads, 1 is tails

		// THE COMPUTER MAKES ITS CHOICE

		// If there's not enough history, randomly choose between 0 and 1,
		// otherwise, choose using a Markov Chain
		computerChoice = (history.length < 3) ? randomInt(2) : markovChain();

		// Save the result of the current round
		chosen = face;
		history.push([chosen, playerWins()])

		if (playerWins()) {
			playerWinCount++;
		} else {
			computerWinCount++;
		}
	}

	// Choose using Markov Chain
	function markovChain() {
		const chunked: Triple[] = history.map((value, idx, arr) => [arr[idx - 2], arr[idx - 1], value]).slice(2) as Triple[];
		// Get the current situation

		const situation: Situation = history.slice(-2) as Situation;
		// Filter the situations that match the current situation
		const filtered: Triple[] = chunked.filter(value => isSame(value, situation));

		if (filtered.length === 0) {
			return randomInt(2);
		}

		const i = randomInt(filtered.length - 1);
		const prediction: Triple = filtered[i];

		// For now, we're going to assume that the computer is Even, which means that it will try to match the player
		return prediction[1][0];
	}

	function playerWins(): boolean {
		if (computerChoice === chosen) {
			return false;
		} else {
			return true;
		}
	}

	// Get a random number between 0 and an upper bound.
	function randomInt(upper: number): number {
		return Math.floor(Math.random() * upper)
	}

	// They are the same situation if in both, for example, the player won, played the same, and won
	function isSame(r1: Triple, r2: Situation): boolean {
		const playedSame1 = r1[0][0] === r1[1][0];
		const playedSame2 = r2[0][0] === r2[1][0];

		return playedSame1 === playedSame2 && r1[0][1] === r2[0][1] && r1[1][1] === r2[1][1];
	}

</script>
<div>
	<p>This is the game of <a href="https://en.wikipedia.org/wiki/Matching_pennies">matching pennies</a>.</p>
	<p>
		In matching pennies, one player is Even and the other is Odd. 
		Both players simultaneously choose either heads or tails for their pennies. If both players chose the same, then the Even player wins. Otherwise, the Odd player wins.
	</p>
	<p>
		This program is based on the Mind-Reading Machine created by Claude Shannon in 1953. It might win so often that it seems like it's cheating. 
		In fact, the program does not use information about your current choice to determine its choice. Instead, it predicts your choice using a Markov Chain simulation.
	</p>
	<p>The computer is Even.</p>
</div>
<div>
	{#if chosen !== -1}	
		<p>You chose {chosen === 0 ? "heads" : "tails"}.</p>
		<p>The computer chose {computerChoice === 0 ? "heads" : "tails"}.</p>
		<p>The player {playerWins() ? "wins" : "loses"}.</p>
	{/if}
</div>
{#if chosen !== -1}
<div>Play again?</div>
{/if}
<div class="counter">
	<button onclick={() => choose(0)} aria-label="Choose heads">
		<h1>H</h1>
	</button>

	<button onclick={() => choose(1)} aria-label="Choose tails">
		<h1>T</h1>
	</button>
</div>
<div class="below-btns">Click H for heads or T for tails.</div>
<p class="below-btns">You're <b>{playerWinCount}-{computerWinCount}</b> against the computer.</p>

<style>
	.below-btns {
		text-align: center;
	}

	.counter {
		display: flex;
		border-top: 1px solid rgba(0, 0, 0, 0.1);
		border-bottom: 1px solid rgba(0, 0, 0, 0.1);
		justify-content: center;
		margin: 1rem 0;
	}

	.counter button {
		width: 2em;
		padding: 0;
		padding-top: 1.2rem;
		display: flex;
		align-items: center;
		justify-content: center;
		border: 0;
		background-color: transparent;
		touch-action: manipulation;
		font-size: 2rem;
	}

	.counter button:hover {
		background-color: var(--lightAccent);
	}

</style>
