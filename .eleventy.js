// Code Highlighting Plugin
const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");

// Render Math
// eleventy-plugin-mathjax was duplicating equations?
// @sunt-programator/eleventy-plugin-mathjax
const mathjaxPlugin = require("eleventy-plugin-mathjax");

// Formatting Dates and fixing off by one error
const {DateTime} = require("luxon")

// Set class attibutes in markdown
const markdownIt = require('markdown-it')
const markdownItAttrs = require('markdown-it-attrs')
const markdownItOptions = {html: true, breaks: true, linkify: true}
const markdownLib = markdownIt(markdownItOptions).use(markdownItAttrs)

module.exports = function (eleventyConfig) {

    // Tell eleventy to grab and watch css
    eleventyConfig.addPassthroughCopy("./src/css")
    eleventyConfig.addWatchTarget("./src/css")

    // Grab the assets when making website
    eleventyConfig.addPassthroughCopy("./src/assets")

    // Use code hightlighting plugin
    eleventyConfig.addPlugin(syntaxHighlight);

    // Use math plugin
    eleventyConfig.addPlugin(mathjaxPlugin);

    // Use Javascript to format dates better
    eleventyConfig.addFilter("postDate", (dateObj) => {
        return DateTime.fromJSDate(dateObj).toLocaleString(DateTime.DATE_FULL);})

    // Get the current year for copyright
    eleventyConfig.addShortcode("year", () => `${new Date().getFullYear()}`);
    
    // Pass the markdown attributes to 11ty
    eleventyConfig.setLibrary('md', markdownLib)

    return {
        dir: {
            input: "src",
            output: "public"
        }
    }
}