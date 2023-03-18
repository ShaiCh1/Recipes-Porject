The goal :
Build an API which receives a URL and returns a json containing the ingredients , the recipes that show up in the web page, and approximates the overall nutritional value of the dish ( amount of sugar, protein, calories, etc.. ).
Assumptions:
- You can assume that all the URLs received are valid
- not every paragraph in the website is relevant to the information you are trying to extract

Guide lines:
- Please provide a full analysis of the running time and memory consumption of your implementation.
- The implementation will be done in python 3.6 or higher.
- Python packages in this assignment: pytorch, tensorflow, pandas, numpy and BeautifulSoup
- Any additional libraries must be approved.
- attached to this document is a small dataset scraped from an example domain, and a dataset that includes nutritional values of food items. You can use additional datasets that are available across the internet. List any additional datasets you used.
- You are advised to look at this as a classification problem that determines for each paragraph with what probability it's label is 'ingredients' or 'recipe', or ' None'
- The code must be managed in git
- Create a run.sh file which accepts a url file and outputs the requested json. - To test the script random valid urls will be chosen on which the model will be run.
- Bonus: have the code run on a Docker.










Example:
Url: https://www.loveandlemons.com/homemade-pasta-recipe/ 
{
Recipe:
[
2 cups all-purpose flour,
3 large eggs,
½ teaspoon sea salt,
½ tablespoon extra-virgin olive oil
],
                                           Nutritional information :
{
		Sugar: X,
		Calories:
		Protein: Z,
		…
}
INSTRUCTIONS:
"Place the flour on a clean work surface and make a nest. Add the eggs, olive oil, and salt to the center and use a fork to gently break up the eggs, keeping the flour walls intact as best as you can. Use your hands to gently bring the flour inward to incorporate. Continue working the dough with your hands to bring it together into a shaggy ball.
Knead the dough for 8 to 10 minutes. At the beginning, the dough should feel pretty dry, but stick with it! It might not feel like it's going to come together, but after 8-10 minutes of kneading, it should become cohesive and smooth. If the dough still seems too dry, sprinkle your fingers with a tiny bit of water to incorporate. If it's too sticky, dust more flour

onto your work surface. Shape the dough into a ball, wrap in plastic wrap, and let rest at room temperature for 30 minutes. Dust 2 large baking sheets with flour and set aside.
Slice the dough into four pieces. Gently flatten one into an oval disk. Run the dough through the Pasta Roller Attachment three times on level 1 (the widest setting).
Set the dough piece onto a countertop or work surface. Fold both short ends in to meet in the center, then fold the dough in half to form a rectangle (see photo above). Run the dough through the pasta roller three times on level 2, three times on level 3, and one time each on levels 4, 5, and 6.
Lay half of the pasta sheet onto the floured baking sheet and sprinkle with flour before folding the other half on top. Sprinkle more flour on top of the second half. Each side should be floured so that your final pasta noodles won't stick together.
Repeat with remaining dough.
Run the pasta sheets through the Pasta Cutter Attachment (pictured is the fettuccine cutter). Repeat with remaining dough. Cook the pasta in a pot of salted boiling water for 1 to 2 minutes.”
}
