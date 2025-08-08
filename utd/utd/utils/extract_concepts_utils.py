import re


def get_prompts_and_examples(concept):
    if concept == 'objects':
        max_new_tokens = 100
        system_prompt = 'You are an intelligent chatbot designed to extract requested information from the textual description of an image.'
        main_prompt_template = ('I will give you a textual description of the image. '
                                'List ALL objects visible in the image. '
                                'An object is anything that has a fixed shape or form, that you can touch or see.  '
                                'Name each object with one noun or a maximum of two words. '
                                'Skip uncertain objects. '
                                'The textual description of the image: "{}" '
                                'DO NOT PROVIDE ANY EXTRA INFORMATION ABOUT OBJECT PROPERTIES OR RELATIONSHIPS TO OTHER OBJECTS IN PARENTHESES. '
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. ")
        text_prompt = "{}"
        answer_prompt = 'Сomprehensive enumerated list of objects:'  # 'The ALL objects visible in the image are:' # was Сomprehensive

        EXAMPLES = [
            (
                "In the photo, there are two people, a man and a woman, who are performing a dance routine together. The woman is standing on the man's shoulders, and they are both in the middle of a dance move. The man is wearing a tie, which adds a touch of formality to their performance. The scene captures the dynamic and energetic nature of their dance, showcasing their skill and coordination.",
                "\n1. Man\n2. Woman\n3. Tie\n",
            ),
            (
                "In the photo, there is a man standing in front of a large machine, which appears to be a factory or industrial setting. The man is wearing a plaid shirt and is positioned in the center of the image. The machine is located on the right side of the man, occupying a significant portion of the background. The scene suggests that the man is either working or observing the machine, possibly as an employee or visitor to the facility.",
                "\n1. Man\n2. Large machine \n3. Plaid shirt\n",
            ),
            (
                'In the photo, we see a scene from a movie or TV show featuring actors portraying characters in a pharmacy setting. The central focus is on two characters:\n\n1. The man on the left is wearing a yellow shirt with a name tag that reads "Rex." He is looking at the woman with a concerned or questioning expression.\n\n2. The woman on the right is wearing a red jacket and is holding a white object, which appears to be a Wii remote, suggesting she might be playing a video game. She is looking at the man with a slightly puzzled or surprised expression.\n\nIn the background, there are shelves stocked with various products, including what looks like pharmaceuticals and possibly some snacks or drinks. The setting is typical of a pharmacy, with a sign that reads "RUGS" and "PHARMACY" visible above the shelves.\n\nThe relationship between the objects in the photo is that they are',
                '\n1. Man\n2. Yellow shirt\n3. Name tag\n4. Woman\n5. Red jacket\n6. White object\n7. Shelves\n8. Various products.\n9. Sign "RUGS”.\n10. Sign "PHARMACY"\n'
            )
        ]
    elif concept == 'activities':
        max_new_tokens = 150
        system_prompt = 'You are an intelligent chatbot designed to extract requested information from the textual description of an image.'
        main_prompt_template = ('I will give you a textual description of the image. '
                                'List all VISIBLE activities in the image. '
                                'Activity is lively action or movement. '
                                'Name each activity with a concise phrase '
                                'SKIP possible or implied activities that are not visible. '
                                'If no activity is visible, reply "No activity is visible." '
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
                                'The textual description of the image: "{}" '
                                )
        text_prompt = "{}"
        answer_prompt = 'Сomprehensive enumerated list of activities:'

        EXAMPLES = [
            (
                "In the photo, we see a young woman standing in a hallway with lockers in the background. She is smiling and appears to be in a good mood. The objects in the photo are arranged in a way that suggests a typical school environment.\n\nThe woman is the central figure in the photo, and her position in the foreground draws the viewer's attention. She is wearing a red and black patterned sweater, which stands out against the more neutral colors of the hallway.\n\nBehind her, there are lockers, which are common in school settings. These lockers are typically used by students to store their belongings. The lockers are arranged in a row, suggesting an organized and structured environment.\n\nOn the right side of the photo, there is a bulletin board with various items pinned to it. This is a common place for school announcements, notices, and decorations. The bulletin board is located next to the lockers",
                "\n1. A woman is standing.\n2. A woman is smiling.\n"
            ),
            (
                "In the photo, there are several objects and their relationships:\n\n1. The person in the foreground is looking at the contents of a locker.\n2. The locker has a red and white sticker on it, which is partially obscured by the person's head.\n3. The person is wearing a green jacket with a red collar, and their hair is pulled back into a ponytail.\n4. In the background, there is another person who appears to be reaching for something on a shelf or in a locker.\n5. The person in the background is wearing a red jacket and has a backpack on their back.\n6. The setting appears to be an indoor environment, possibly a school or a similar institution, given the presence of lockers.\n\nThe relationships between these objects suggest a casual, everyday scene where individuals are interacting with their personal belongings. The person in the foreground is focused on their own locker,",
                "\n1. The person is looking in the locker.\n2. Another person is reaching for something on a shelf or in a locker.\n"),
            (
                "The image appears to be a still from an animated television show or film. It features two characters: a snail and a purple creature with a shell. The snail is in the foreground, with its head and tentacles prominently displayed. The purple creature is in the background, with its shell and tentacles visible. The snail and the purple creature are positioned close to each other, suggesting a relationship or interaction between them. The background is a simple, blue, underwater environment with a few bubbles, which is typical for scenes set in the ocean. The characters' expressions and body language could provide more context about their relationship, but without additional information, it's not possible to determine the exact nature of their interaction.",
                "\nNo activity is visible.\n"
            )
        ]
    elif concept == 'verbs':
        max_new_tokens = 150

        system_prompt = 'You are an intelligent chatbot designed to extract requested information from the textual description of an image.'

        main_prompt_template = ('I will give you a list of visible activities of the image. '
                                'You task is to delete information about objects from this description. '
                                'Replace all objects in this list with "someone" or "something," but keep the activity.'
                                'If you have to, you may delete some details, but delete ALL object information. '
                                'If the input is "No activity is visible.", keep it "No activity is visible."'
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
                                'The list of visible activities: "{}" '
                                )
        text_prompt = "{}"
        answer_prompt = 'Post-processed enumerated list of activities:'

        EXAMPLES = [
            (
                "1. Two hands are placed on a grid.\n2. The hands are positioned with fingers spread out and palms down.\n3. The hands are aligned along the grid lines.\n\nThese actions suggest that the individuals are using the grid for measuring or calibration purposes.\n",
                "1. Something is placed on something.\n2. Something is positioned.\n3. Something is aligned along something.\n"
            ),
            (
                "1. The baseball player is walking.\n2. The spectators are standing.\n",
                "1. Someone is walking.\n2. Someone is standing.\n"
            ),
            (
                "1. A human hand is cradling a figurine.\n2. Soap is being used.\n3. A figurine is being bathed.\n",
                "1. Something is cradling something.\n2. Something is being used.\n3. Something is being bathed.\n"
            ),
            (
                "1. The people are engaged in a lively conversation or event.\n",
                "1. Someone is talking or engaged into something.\n"
            ),
            (
                "1. Vehicles are moving along the road.\n2. Traffic lights are controlling the flow of vehicles.\n\n",
                "1. Something is moving.\n2.Something is controlling something.\n"
            )
        ]
    elif concept == 'objects+composition+activities_15_words':
        max_new_tokens = 1000

        system_prompt = 'You are an intelligent chatbot designed to extract requested information from the textual description of an image.'

        main_prompt_template = ('Summarize the following image description in 15 words: "{}" '
                                )
        text_prompt = "{}"
        answer_prompt = '15-words summary:'
        EXAMPLES = [
        ]
    else:
        raise NotImplementedError
    return system_prompt, main_prompt_template, text_prompt, answer_prompt, EXAMPLES, max_new_tokens


def parse_llm_output(text, concept):
    if concept in ['objects', 'activities', 'verbs']:
        # remove numerations
        pat = re.compile(r'[1-9]*\.(.*)')
        output = []
        for x in text.split('\n'):
            match = pat.match(x)
            if match is not None:
                x = match.group(1)
                # remove parentheses
                x = re.sub(r'\([^)]*\)', '', x)
                x = x.strip()
            output.append(x)
        if concept == 'objects':
            output = ', '.join(output)
            output = output.lower()
        else:
            output = ' '.join(output)
        return output.strip()
    elif concept == 'objects+composition+activities_15_words':
        return text.strip()
    else:
        raise NotImplementedError