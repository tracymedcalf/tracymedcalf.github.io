<script>
	import { siteAuthor } from '$lib/config';
	import NavItems from './NavItems.svelte';
</script>

<footer>
	<nav>
		<ul>
			<li>
				<a href="/">Home</a>
			</li>
		</ul>
	</nav>
	<nav>
		<NavItems />
	</nav>

	<ul>
		<li>&copy;{new Date().getFullYear()} {siteAuthor}</li>
		<li>Favicon by <a href="[URL_TO_AUTHOR_PROFILE]">damaswahyu</a> is licensed under <a href="https://creativecommons.org">CC BY 4.0</a></li>
	</ul>
</footer>
