import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import { mdsvex } from "mdsvex";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeSlug from "rehype-slug";
import remarkMath from "remark-math";
import rehypeKatexSvelte from "rehype-katex-svelte";

const config = {
  extensions: [".svelte", ".md"],
  kit: {
    adapter: adapter(),
    prerender: {
      entries: [
        "*",
        "/api/posts/page/*",
        "/blog/category/*/page/",
        "/blog/category/*/page/*",
        "/blog/category/page/",
        "/blog/category/page/*",
        "/blog/page/",
        "/blog/page/*",
      ] 
    }
  },
  preprocess: [
    mdsvex({
      extensions: [".md"],
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatexSvelte, rehypeSlug, rehypeAutolinkHeadings]
    }),
    vitePreprocess()
  ]
};

export default config;
