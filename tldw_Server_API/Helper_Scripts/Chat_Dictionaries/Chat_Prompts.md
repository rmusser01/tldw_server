# Chat Prompts Dictionary
custom_prompt_summarize_bulleted_notes: |
    You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: 
        Use multiple headings based on the referenced topics, not categories like quotes or terms. 
        Headings should be surrounded by bold formatting and not be listed as bullet points themselves. 
        Leave no space between headings and their corresponding list items underneath. 
        Important terms within the content should be emphasized by setting them in bold font. 
        Any text that ends with a colon should also be bolded. 
        Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. 
        Do not reference these instructions within the notes.```
    Based on the content between backticks create comprehensive bulleted notes.

    **Bulleted Note Creation Guidelines**

    **Headings**:
    - Based on referenced topics, not categories like quotes or terms
    - Surrounded by **bold** formatting 
    - Not listed as bullet points
    - No space between headings and list items underneath

    **Emphasis**:
    - **Important terms** set in bold font
    - **Text ending in a colon**: also bolded

    **Review**:
    - Ensure adherence to specified format
    - Do not reference these instructions in your response.
---@@@---



### Prompts by Tostino - https://gist.github.com/Tostino
tostino_gist_1_system_prompt: |
    Your main objective is to condense the content of the document into a concise summary, capturing the main points and themes.
    Please read the provided Original section to understand the context and content. Use this understanding to generate a summary of the Original section. Separate the article into chunks, and sequentially create a summary for each chunk. Focus on summarizing the Original section, ignoring any details about sponsorships/advertisements in the text.

    Summarized Sections:
    1. For each chunk, provide a concise summary. Start each summary with "Chunk (X of Y):" where X is the current chunk number and Y is the total number of chunks.

    To craft a Final Summary:
    1. Read the Summarized Sections: Carefully review all the summarized sections you have generated. Ensure that you understand the main points, key details, and essential information from each section.
    2. Identify Main Themes: Identify the main themes and topics that are prevalent throughout the summarized sections. These themes will form the backbone of your final summary.
    3. Consolidate Information: Merge the information from the different summarized sections, focusing on the main themes you have identified. Avoid redundancy and ensure the consolidated information flows logically.
    4. Preserve Essential Details: Preserve the essential details and nuances that are crucial for understanding the document. Consider the type of document and the level of detail required to capture its essence.
    5. Draft the Final Summary: After considering all the above points, draft a final summary that represents the main ideas, themes, and essential details of the document. Start this section with "Final Summary:"

    Ensure that your final output is thorough, and accurately reflects the document’s content and purpose.
---@@@---


tostino_gist_2_resummarize_prompt: |
    Your main objective is to condense the content of the document into a concise summary, capturing the main points and themes.
    Please read the provided Original section to understand the context and content. Use this understanding to generate a summary of the Original section, incorporating relevant details and maintaining coherence with the Prior Summary.
    Notes:
    - The Prior Summary was created from the chunk of the document directly preceding this chunk.
    - Ignore the details already included in the Prior Summary when creating the new Summary.
    - Focus on summarizing the Original section, taking into account the context provided by the Prior Summary.
    - Ignore any details about sponsorships/advertisements in the text.
    <#user_context#>
    Prior Summary:
---@@@---




tostino_gist_3_system: |
    Your main objective is to condense the content of the document into a concise summary, capturing the main points and themes.
    To craft a Final Summary:

    1. Read Summarized Sections: Carefully review all the summarized sections of the document. Ensure that you have a clear understanding of the main points, key details, and essential information presented in each section.
    2. Identify Main Themes: As you go through the summarized sections, identify the main themes and topics that are prevalent throughout the document. Make a list of these themes as they will form the backbone of your final summary.
    3. Consolidate Information: Merge the information from the different summarized sections, focusing on the main themes you have identified. Avoid redundancy and ensure that the consolidated information flows logically.
    4. Preserve Essential Details: While consolidating, ensure that you preserve the essential details and nuances that are crucial for understanding the document. Consider the type of document and the level of detail required to accurately capture its essence.
    5. Check for Completeness: After drafting the final summary, review it to ensure that it accurately represents the main ideas, themes, and essential details of the document.

    Please remember to be thorough, and ensure that the final summary is a true reflection of the document’s content and purpose.
---@@@---


tostino_inkbot_chunked_summary: |
    Your main objective is to condense the content of the document into a concise summary, capturing the main points and themes.
    Please read the provided Original section to understand the context and content. Use this understanding to generate a summary of the Original section, incorporating relevant details and maintaining coherence with the Prior Summary.

    Notes:
    - The Prior Summary was created from the chunk of the document directly preceding this chunk.
    - Ignore the details already included in the Prior Summary when creating the new Summary.
    - Focus on summarizing the Original section, taking into account the context provided by the Prior Summary.
    - Ignore any details about sponsorships/advertisements in the text.
    <#user_context#>
    Prior Summary:
    Meituan, China's third-largest tech company, has the majority share of the Chinese online hotel booking market. In addition to its food delivery service, Meituan offers travel and hotel booking products. The company has expanded its services to include supply chain, online advertising tools, and cloud-based enterprise resource planning software for small business merchants. Meituan follows a strategy called the "Food Plus Platform," similar to WeChat's super app model. The food delivery business is crucial for Meituan as it drives traffic and cross-selling opportunities for its other services. Meituan's scale allows for efficient delivery operations and low-cost leadership. Meituan has backing from both Alibaba and Tencent, with Tencent's favorable feature placement within the WeChat ecosystem. Alibaba sold its stake in Meituan, citing its focus on its own online-to-offline services division.

    Original (Chunk 3 of 3 total):
    Alibaba thereafter began developing several directly competing services to Meituan. As Meituan finished up in 2020, they warned investors to expect several quarters of operating losses as the company continues its battles and expands into community buying. To fund this expansion, the company raised $10 billion by selling debt and equity.

    Wang is so willing to stir up dust with titans like Alibaba because he is, in general, very comfortable with competition. Meituan's style of horizontal expansion means charging forward into a field of crowded incumbents. This latest competitive push takes Meituan into community purchasing with its Meituan Select service. Community purchasing is a new Chinese e-commerce trend where communities can set up local groups for bulk buying. It's kind of like Groupon but more localized. Pinduoduo pioneered this model where people can band together to unlock group discounts. Founded in 2015, Pinduoduo is now worth $140 billion and its founder Colin Huang is one of China's richest people. The community group buying concept began in China's hinterlands, massive cities with millions of people that nobody has ever heard of, like Dandong, Panjing, and Fushun. It's a titanic market estimated to be worth $100 billion in 2021. Meituan is not going to win it though without a fight. Meituan Select directly competes with JD.com's Dingdong Maicai, Xingshen Youxuan, Pinduoduo's Duoduo Maicai, Alibaba's Taobao Maicai, and Didi Chuxing's Chengxin Youxuan. I should mention, interestingly enough, Pinduoduo is also a Tencent investment. Having scaled up in their early days due to their usage of WeChat groups and mini-programs, the Meituan team is definitely not afraid of stepping on their siblings' toes. Wang is a student of history. In interviews, he quotes Chairman Mao and cites the Hundred Regiments Offensive, a glorious WWII Chinese victory commanded by CCP Kuomintang General Peng Dehuai. He would be a great astronomy viewer. Considering Wang's knowledge of history, a recently deleted post on his social media is worth closing with. In May 2021, he posted on the social media network fanfo.com a 1,000-year-old poem by Zhang Jie titled "The Book-Burning Pit". Then he deleted it because he knew he'd done goofed. Meituan's stock crashed to a seven-month low as the market immediately saw the post as a criticism of Beijing and its recent antitrust moves. The Chinese government has been very sensitive to such things, starting with the very public beatdown of Alibaba Group after the Jack Ma speech. Meituan is very comfortable with conflict. They have carried their zen through years of battles and horizontal industry invasions. But now perhaps they have come across an opponent against which they cannot win. Alright, everyone, that is it for tonight. Thanks for watching. If you want more content, you can like and subscribe to the channel. I would like if you did. Remember to sign up for the email newsletter. And if you want to send me an email, drop me a line at john@asianometry.com. I love getting letters from viewers. Until next time, I will see you guys later.

    Summary (Chunk 3 of 3 total):
    <#bot#>
    Meituan, China's third-largest tech company, is expanding into community buying with its Meituan Select service, competing with existing players like JD.com's Dingdong Maicai, Xingshen Youxuan, Pinduoduo's Duoduo Maicai, Alibaba's Taobao Maicai, and Didi Chuxing's Chengxin Youxuan. Despite having a strong presence in various industries, Meituan may face challenges in winning over the community buying market. Wang, the CEO of Meituan, is known for his comfort with competition and his historical knowledge, often referencing Chairman Mao and the Hundred Regiments Offensive. However, Meituan's recent post on social media criticizing Beijing's antitrust moves caused their stock to crash, indicating potential sensitivity from the government.
---@@@---


tostino_inkbot_summary_of_summaries: |
    Your main objective is to condense the content of the document into a concise summary, capturing the main points and themes.
    To craft a Final Summary:

    1. Read Summarized Sections: Carefully review all the summarized sections of the document. Ensure that you have a clear understanding of the main points, key details, and essential information presented in each section.
    2. Identify Main Themes: As you go through the summarized sections, identify the main themes and topics that are prevalent throughout the document. Make a list of these themes as they will form the backbone of your final summary.
    3. Consolidate Information: Merge the information from the different summarized sections, focusing on the main themes you have identified. Avoid redundancy and ensure that the consolidated information flows logically.
    4. Preserve Essential Details: While consolidating, ensure that you preserve the essential details and nuances that are crucial for understanding the document. Consider the type of document and the level of detail required to accurately capture its essence.
    5. Check for Completeness: After drafting the final summary, review it to ensure that it accurately represents the main ideas, themes, and essential details of the document.

    Please remember to be thorough, and ensure that the final summary is a true reflection of the document’s content and purpose.

    Summarized Sections:
    Chunk (1 of 3):
    The Franco-Prussian War of 1870-71 is about to begin due to the growing rivalry between the French Empire and the German Kingdom of Prussia. Prussian Prime Minister Otto von Bismarck dreams of a united Germany under Prussian leadership and sees the war as an opportunity to bind the southern German states to Prussia. French Emperor Napoleon III and Empress Eugénie also have dreams of a glorious French Empire and see the rise of Prussia as a threat. The crisis that triggers the war begins in Spain, where Leopold von Hohenzollern-Sigmaringen is chosen to be the new Spanish king. This news is seen as explosive in Paris, as it is unacceptable to the French government that the Prussian royal house might also rule Spain. French Foreign Minister the Duc de Gramont sees this as an opportunity to humiliate Prussia or win a war against them. He demands that Prussia force Leopold to refuse the Spanish offer, but Prussian King Wilhelm I refuses to make the declaration. This leads to the Ems Dispatch, a manipulated message by Bismarck that reaches French newspapers on July 14, shocking the French government and pushing them towards war.

    Chunk (2 of 3):
    On July 14th, the Council of Ministers meets to discuss the demands made by French Foreign Minister Gramont to the Prussians. The ministers, who had not approved these demands, are divided but eventually agree to Emperor Napoleon III's proposal for an international conference to resolve the crisis and avoid war. However, Empress Eugénie and War Minister Le Boeuf pressure Napoleon III to scrap the conference and mobilize the army, and he gives in. This informal declaration of war against Prussia leads to a heated debate in the French Parliament the next day, where members ultimately vote to declare war and provide war credits. The French population is described as being in a state of fervor and excitement, with bands marching through the streets and people expressing their enthusiasm for war. In response, Prussia and its allies also mobilize their armies. In Bavaria, there is a debate among lawmakers about whether to join the war, with concerns raised about the safety of the Palatinate region. Some lawmakers argue that the Spanish throne question has nothing to do with Germany and that they do not want their region to suffer the horrors of war.

    Chunk (3 of 3):
    Bavaria and the southern kingdom of Württemberg are hesitant about joining the war, fearing the pacification of their kingdoms and jeopardizing their good relations with France. However, both kingdoms ultimately mobilize their armies. The Grand Duchy of Baden, despite having good relations with Prussia, decides to mobilize as they believe only Prussia can protect them in the event of war. With the mobilization of the Grand Duchy of Hesse, Prussian King Wilhelm I now commands a full German army, but there is still no official declaration of war. The Ems Dispatch has given the war proponents in France and Prussia the excuse they need, with Empress Eugénie and Bismarck using the war to achieve their respective goals. Next week, the Franco-Prussian War will officially begin, marking the start of a new Prussian Empire.

    The Franco-Prussian War of 1870-71 was instigated by the growing tension between France and Prussia, fueled by the latter's ambition to unite Germany under its rule and the former's desire to maintain its own imperial dominance. The conflict began when the French government objected to the candidacy of Leopold von Hohenzollern-Sigmaringen as the new Spanish king, which they saw as a threat to their interests. This led to a series of events that culminated in the Ems Dispatch, a manipulated message from Prussian Prime Minister Otto von Bismarck, which incensed the French government and pushed them towards war. Despite initial plans for an international conference to resolve the crisis, French Empress Eugénie and War Minister Le Boeuf pressured Emperor Napoleon III to mobilize the army, leading to an unofficial declaration of war. Both France and Prussia mobilized their armies, with some German kingdoms like Bavaria and Württemberg initially hesitant but ultimately joining the conflict. The Grand Duchy of Baden joined out of fear of Prussian retaliation, while the Grand Duchy of Hesse did so due to its belief in Prussian protection. The Ems Dispatch provided the necessary justification for war proponents in both countries, allowing Empress Eugénie and Bismarck to pursue their individual agendas. Thus, the Franco-Prussian War formally commenced, marking the beginning of a new era for Prussia.
---@@@---






