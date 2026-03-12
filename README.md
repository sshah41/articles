# Technical Blog: Nemotron 3 Super

This repository contains a deep dive technical article on NVIDIA's **Nemotron 3 Super**, exploring its 88-layer hybrid architecture, Mamba-2 sequences, LatentMoE bottlenecks, and multi-token prediction efficiencies that uniquely enable 1 million token context lengths.

## GitHub Pages Deployment

This project is configured to be deployed as a static GitHub Pages site. The root `index.md` file contains the complete markdown rendering of the blog.

### Key Features
*   **Mermaid.js Visualizations:** The blog leverages GitHub's native `mermaid` codeblock support to procedurally generate technical diagrams dynamically (Architecture block, KV Cache state contrasts, and LatentMoE mechanisms) directly from text formatting. No fragile image blobs required.
*   **Minimal Theme:** Designed to be clean and readable for ML engineers and technically minded builders. Emphasizing typography and layout pacing over bloat.

## How to use:
To run this locally, you can use any standard markdown viewer or rely on Jekyll:
```bash
bundle install
bundle exec jekyll serve
```
Alternatively, simply enable GitHub Pages from the repository Settings tab, mapped to the primary branch's `/` root folder.
