from word2world import Config
from word2world import Word2WorldEnv, LLMAgent
from word2world.utils import (
    extract_between_ticks,
    create_colored_tilemap_image,
    create_legend_image,
    create_legend,
    extract_present_elements,
    euclidean_distance,
    extract_list,
    extract_dict,
    get_image_color_tile_mapping,
    list_of_lists_to_string,
    find_character_position,
    overlap_dict,
    find_most_similar_images
    )
from word2world.fixers import remove_extra_special_chars, pad_rows_to_max_length
from word2world.solvers import find_characters, parse_grid, find_important_tiles, EnhancedAStarWorldAgent, WorldState
from .generation_base import Evaluator, Generator


import matplotlib.pyplot as plt
import numpy as np
import random
import json
import traceback
import time
import imageio
import pandas as pd
from PIL import Image
import traceback
import transformers
import torch
import os
import tempfile

cfg = Config()

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

model = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def change_answer(text):
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix=".txt") as temp_file:
        temp_file.write(text)
        temp_file.flush()
        
        # Open the text in the default editor
        os.system(f'{os.getenv("EDITOR", "nano")} {temp_file.name}')
        
        # Read the edited text back from the file
        temp_file.seek(0)
        user_text = temp_file.read()
    
    # Delete the temporary file
    os.remove(temp_file.name)
    
    # Print the final text
    print(f"Final text: {user_text}")
    return user_text


class LlamaEvaluator(Evaluator):
    def __init__(self, total_input_tokens, total_output_tokens):
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        super().__init__(self.model_id, total_input_tokens, total_output_tokens)

    def getTokensSize(self, msg):
        return len(tokenizer.encode(msg, add_special_tokens=False))

    
    def evaluate_world(self, map, tile_map_dictionary, story, walkable_tiles, important_tiles, previous_maps):
        print(f"Evaluating World...")    
        no_of_exceptios = 0
        eval_system_prompt = "You are an evaluator of a 2D tilemap world created from a story. You extract meaning from a given story. You are also provided by Python dictionary-like mapping of tiles and characters or alphabets. You evaluate based on how the tiles have been placed and if they match how the story has explained. Your result being 'No' is not a bad thing, but it actually gives a good insight. answere only 'Yes' or 'No'"
        evaluation_prompt = f"Given the story, tiles used to create the 2D map, and the 2D map, suggest whether the 2D tilemap world is coherent to the story. The story is as follows:\n{story}\nThe tile mapping:{tile_map_dictionary}\nThe 2D tilemap world:\n{map}\n Check for all the tiles mentioned in the tile mapping being in the 2D tilemap world. Strictly return only 'Yes' or 'No' in your answer. dont answere anything else dont give me code and dont write sentances! you answere should be only 'Yes' or 'No'"
        done = False
        while not done:
            try:
                print(f"check#1 done = {done}")
                
                world_eval = model([
                                                    {"role": "system", "content": eval_system_prompt},
                                                    {"role": "user", "content": evaluation_prompt}
                                                 ],
                                        max_new_tokens=1500,
                                        temperature = 0.6)[0]["generated_text"][-1]["content"]
                print("world eval:")
                print(world_eval)
                input_message = eval_system_prompt + "\n" + evaluation_prompt
                input_tokens = self.getTokensSize(input_message)
                compeltion_tokens = self.getTokensSize(world_eval)
                
                self.total_input_tokens.append(input_tokens)
                self.total_output_tokens.append(compeltion_tokens)
                
                world_eval_dictionary = extract_dict(world_eval)    
                world_eval_dictionary = self.tile_accuracy(map, world_eval_dictionary, important_tiles)
                world_eval_dictionary = self.euclidean_distance(map, previous_maps, world_eval_dictionary)

                print("Evaluation: \n", world_eval_dictionary, "\n")
    
                done = True


            except Exception as e:
                tb = traceback.format_exc()
                print(f"Exception raised: {e}\n{tb}")
                no_of_exceptios += 1
                if no_of_exceptios >= 5:
                    done = True
                pass
        
        return world_eval_dictionary, self.total_input_tokens, self.total_output_tokens
    

class LlamaGenerator(Generator):
    def __init__(self, total_input_tokens, total_output_tokens):
        self_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        super().__init__(self_id, total_input_tokens, total_output_tokens)
    
    def getTokensSize(self, msg):
        return len(tokenizer.encode(msg, add_special_tokens=False))

    def create_story(self, story_paragraphs, total_objectives):
        print("Creating story...")
        print(f"Number of story paragraphs:{story_paragraphs}, Objectives of story: {total_objectives}")
        story_prompt = f"Write a {story_paragraphs[0]}-{story_paragraphs[1]} paragraph story which has characters including the protagonist trying to achieve something and the antagonist wanting to stop the protagonist. The story should describe an environment(s) where the story is set up. There should be {total_objectives} objectives for the protagonist in the story. One of them should be to defeat the antagonist somehow."
        story = model([{"role": "user", "content": story_prompt}], max_new_tokens=1500)[0]["generated_text"][-1]["content"]
        input_tokens = self.getTokensSize(story_prompt)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(self.getTokensSize(story))
        print("\n")
        print(story)
        print("\n")

        return story, story_prompt

    def extract_character_info(self, story, story_prompt):
        print("Extracting character information...")
        
        character_prompt = "Let's use the above story to create a 2D game. Write a specific description of each character which can be used as a prompt to generate sprites for the characters."
        character_discriptions = model([
                                                                           {"role": "user", "content": story_prompt},
                                                                           {"role": "assistant", "content": story},
                                                                           {"role": "user", "content": character_prompt},
                                                                       ], 
                                                              max_new_tokens=1500,
                                                              temperature = 1)[0]["generated_text"][-1]["content"]
        input_prompt = story_prompt + "\n" + story + "\n" + character_prompt
        input_tokens = self.getTokensSize(input_prompt)
        output_tokens = self.getTokensSize(character_discriptions)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(output_tokens)
        print("\n")
        print(character_discriptions)
        print("\n")
        
        character_dict_prompt = """Let's use the above story and charecter descriptions to create a 2D game. Create a Python dictionary that has keys as 'Protagonist', 'Antagonist' and any other character and values as very precise description. For example, the dictionary should look like:
        {
            'Protagonist': 'red dressed girl'
            'Antagonist': 'blue dressed boy'
        }
        The description should only be like this. Do not return in a Python response, and do not return anything else but the answere. make the answere look exactly like the example because we are going to use it with a script
        """
        character_dict_discriptions = model([
                                                            {"role": "user", "content": story_prompt},
                                                            {"role": "assistant", "content": story},
                                                            {"role": "user", "content":character_prompt},
                                                            {"role": "assistant", "content": character_discriptions},
                                                            {"role": "user", "content": character_dict_prompt}
                                                          ], 
                                                 max_new_tokens=1500,
                                                temperature = 1)[0]["generated_text"][-1]["content"]
        input_prompt = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions  + "\n" + character_dict_prompt
        input_tokens = self.getTokensSize(input_prompt)
        output_tokens = self.getTokensSize(character_dict_discriptions)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(output_tokens)
        print("\n")
        print(character_dict_discriptions)
        print("\n")

        antagonist_name = ""
        protagonist_name = ""
        character_dict_discriptions = change_answer(character_dict_discriptions)
        character_discriptions_dict = extract_dict(character_dict_discriptions)
        return character_discriptions, character_discriptions_dict, character_prompt, protagonist_name, antagonist_name

    def extract_tileset_info(self, story, story_prompt, character_discriptions, character_prompt):
        print("Extracting tileset information...")
        tileset_prompt = "Create an exhaustive list of tiles needed to create the environment."
        tileset_discriptions = model([
                                                        {"role": "user", "content": story_prompt},
                                                        {"role": "assistant", "content": story},
                                                        {"role": "user", "content": character_prompt},
                                                        {"role": "assistant", "content": character_discriptions},
                                                        {"role": "user", "content": tileset_prompt}
                                                   ], 
                                                max_new_tokens=1500,
                                                temperature = 1)[0]["generated_text"][-1]["content"]
        input_prompt = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions + "\n" + tileset_prompt
        input_tokens = self.getTokensSize(input_prompt)
        output_tokens = self.getTokensSize(tileset_discriptions)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(output_tokens)
        print("\n")
        print(tileset_discriptions)
        print("\n")

        return tileset_discriptions, tileset_prompt

    def map_tiles_to_chars(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt):
        print("Mapping tiles to characters...")
        tileset_map_prompt = "Create a Python dictionary where each unique environment tile or character is mapped to a distinct char. Each of the environment tiles should be represented by only one alphabetic char(A-Z or a-z), while character tiles should be represented by only one special char (e.g., @, #, $, %, etc.), ignore other tiles which are not enviroment or charcters. If there are more enviroment tiles then what you can represent with the alphabetic chars the reduce the number of tiles (max of enviroment tiles is 52). Each tile must is represented by a different tile. Notice that you gave us characters and tiles in the previous messages. In the end, review the answer and fix it if needed. If there are different tiles that are represented by the same key, change it. The protagonist should be '@', and the antagonist '#'. Only provide the dictionary as the output in the format: { 'Protagonist': '@', 'Antagonist': '#', 'TileName': 'Char' }. Provide only the dictionary in your response without any additional text. Make sure it is not returned in a Python response. Names should be the Keys and chars should be the Values. Ensure that only characters are described by one special char, and environments are described by one alphabet char. make sure to check yourself before answering. Example for the format: { 'Protaganist': '@', 'antagonist': '#', 'lora': '+', 'sand': 'a', 'rock': 'b' }"


        # tileset_map_prompt = "Imagine each tile maps to an alphabet or a character. For environment use alphabets and for characters use special characters. Create it in a single Python Dictionary style. Return only and only a Python Dictionary and nothing else in your response. Don't return it in a Python response. Names should be the Keys and alphabets or characters should be the Values. Protagonist should always strictly be '@' and the antagonist should always strictly be '#'."
        tileset_map_discriptions = model([
                                                        {"role": "user", "content": story_prompt},
                                                        {"role": "assistant", "content": story},
                                                        {"role": "user", "content": character_prompt},
                                                        {"role": "assistant", "content": character_discriptions}, 
                                                        {"role": "user", "content": tileset_prompt},
                                                        {"role": "assistant", "content": tileset_discriptions},
                                                        {"role": "user", "content": tileset_map_prompt}
                                                       ],
                                              max_new_tokens=1500,
                                              temperature = 1)[0]["generated_text"][-1]["content"]
        
        input_prompt = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions + "\n" + tileset_prompt + "\n" + tileset_discriptions + "\n" + tileset_map_prompt
        input_tokens = self.getTokensSize(input_prompt)
        output_tokens = self.getTokensSize(tileset_map_discriptions)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(output_tokens)
        print("\n")
        tileset_map_discriptions = change_answer(tileset_map_discriptions)
        tile_map_dict = extract_dict(tileset_map_discriptions)
        print(tileset_map_discriptions)
        print("\n")

        return tile_map_dict, tileset_map_discriptions, tileset_map_prompt

    def extract_goals(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt, tileset_map_discriptions, tileset_map_prompt):
        print("Extracting goals..")
        goal_prompt = "What is the main goal for protagonist of the story? What are the small goals for protagonist to achieve the main goal of the story? Also create rewards and penalties based on the goals for protagonist. Create a score for each reward or penalty"
        goal_discriptions = model([
                                                    {"role": "user", "content": story_prompt},
                                                    {"role": "assistant", "content": story},
                                                    {"role": "user", "content": character_prompt},
                                                    {"role": "assistant", "content": character_discriptions},
                                                    {"role": "user", "content": tileset_prompt},
                                                    {"role": "assistant", "content": tileset_discriptions},
                                                    {"role": "user", "content": tileset_map_prompt},
                                                    {"role": "assistant", "content": tileset_map_discriptions},
                                                    {"role": "user", "content": goal_prompt}
                                                ], 
                                                max_new_tokens=1500,
                                                temperature = 1)[0]["generated_text"][-1]["content"]
        
        input_prompt = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions + "\n" + tileset_prompt + "\n" + tileset_discriptions + "\n" + tileset_map_prompt + "\n" + tileset_map_discriptions + "\n" + goal_prompt
        input_tokens = self.getTokensSize(input_prompt)
        output_tokens = self.getTokensSize(goal_discriptions)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(output_tokens)
        print("\n")
        print(goal_discriptions)
        print("\n")

        return goal_discriptions, goal_prompt

    def extract_important_tiles(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt, tile_map_dict, tileset_map_discriptions, tileset_map_prompt, goal_discriptions, goal_prompt):
        print("Extracting important tiles..")
        important_tile_prompt = f"Considering the above goals that you extracted from the story and given the tile set: {tile_map_dict}\n, create a Python list containing only the chars of the 15 most important tiles that should be placed on the 2D tile map. The list must include the protagonist '@', antagonist '#', and any non-player characters if they exist (described by special char). Only return the list of chars that describes the charcters and the tiles without any additional text or full tile names. The output should be simillar to this format: ['a', 'b', '@', '#'] (including 15 chars that describes the most important tiles). Ensure the list contains only chars."


        # important_tile_prompt = f"Considering the above goals that you extracted from the story and the following tileset\n{tile_map_dict}\n create a Python list of the 15 most important chars of the tiles that should be placed in the 2D tilemap world. Remember, Protagonist, antagonist and non-player characters, if there are any, in the story will always be an important tiles. Only return the important tiles. dont write anything else but the list example for answere: ['a','b','@','#']. dont put in any index in the list the full name of the tile onlt use the chat that decribes the tile"
        important_tile_discriptions = model([
                                                            {"role": "user", "content": story_prompt},
                                                            {"role": "assistant", "content": story},
                                                            {"role": "user", "content": character_prompt},
                                                            {"role": "assistant", "content": character_discriptions},
                                                            {"role": "user", "content": tileset_prompt},
                                                            {"role": "assistant", "content": tileset_discriptions},
                                                            {"role": "user", "content": tileset_map_prompt},
                                                            {"role": "assistant", "content": tileset_map_discriptions},
                                                            {"role": "user", "content": goal_prompt},
                                                            {"role": "assistant", "content": goal_discriptions},
                                                            {"role": "user", "content": important_tile_prompt}
                                                          ], 
                                                  max_new_tokens=1500,
                                                  temperature = 1)[0]["generated_text"][-1]["content"]
        
        input_prompt = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions + "\n" + tileset_prompt + "\n" + tileset_discriptions + "\n" + tileset_map_prompt + "\n" + tileset_map_discriptions + "\n" + goal_prompt + "\n" + goal_discriptions + "\n" + important_tile_prompt
        input_tokens = self.getTokensSize(input_prompt)
        output_tokens = self.getTokensSize(important_tile_discriptions)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(output_tokens)
        print("\n")
        print(important_tile_discriptions)
        important_tile_discriptions = change_answer(important_tile_discriptions)
        important_tiles_list = extract_list(important_tile_discriptions)
        print(important_tiles_list)
        print("\n")

        return important_tiles_list, important_tile_discriptions, important_tile_prompt

    def extract_walkable_tiles(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt, tile_map_dict, tileset_map_discriptions, tileset_map_prompt, goal_discriptions, goal_prompt):
        print("Extracting walkable tiles..")
        walkable_tile_prompt = f"Given the goals extracted from the story and the following tileset: {tile_map_dict}, create a Python list containing only the chars representing the walkable tiles in the 2D tilemap world.\n The answer must be a list containing only the chars of the walkable tiles, without any Python code or additional text. If the output includes anything other than the list of walkable tile characters, correct it so that only the list remains. \n For example, if the environment tiles are: {{'sand':'S', 'pavement':'P', 'grass':'G'}}, the answer should be: ['S', 'P', 'G']."
        walkable_tile_discriptions = model([
                                                            {"role": "user", "content": story_prompt},
                                                            {"role": "assistant", "content": story},
                                                            {"role": "user", "content": character_prompt},
                                                            {"role": "assistant", "content": character_discriptions},
                                                            {"role": "user", "content": tileset_prompt},
                                                            {"role": "assistant", "content": tileset_discriptions},
                                                            {"role": "user", "content": tileset_map_prompt},
                                                            {"role": "assistant", "content": tileset_map_discriptions},
                                                            {"role": "user", "content": goal_prompt},
                                                            {"role": "assistant", "content": goal_discriptions},
                                                            {"role": "user", "content": walkable_tile_prompt}
                                                         ], 
                                                max_new_tokens=1500,
                                                temperature = 1)[0]["generated_text"][-1]["content"]
        input_prompt = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions + "\n" + tileset_prompt + "\n" + tileset_discriptions + "\n" + tileset_map_prompt + "\n" + tileset_map_discriptions + "\n" + goal_prompt + "\n" + goal_discriptions + "\n" + walkable_tile_prompt 
        input_tokens = self.getTokensSize(input_prompt)
        output_tokens = self.getTokensSize(walkable_tile_discriptions)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(output_tokens)
        print("\n")
        print(walkable_tile_discriptions)
        walkable_tile_discriptions = change_answer(walkable_tile_discriptions)
        walkable_tiles_list = extract_list(walkable_tile_discriptions)
        print(walkable_tiles_list)
        print("\n")

        return walkable_tiles_list, walkable_tile_discriptions, walkable_tile_prompt

    def extract_interactive_object_tiles(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt, tile_map_dict, tileset_map_discriptions, tileset_map_prompt, goal_discriptions, goal_prompt):
        print("Extracting walkable tiles..")
        object_tile_prompt = object_tile_prompt = f"Considering the goals extracted from the story and the following tileset: {tile_map_dict}, create a Python list containing only the characters representing the object tiles that can be interacted with in the 2D tilemap world.\n Return only the list of characters for the interactable object tiles, without any additional text or Python code. \n For example, if the object tiles are: {{'door':'D', 'sword':'S'}}, the answer should be: ['D', 'S']."
        object_tile_discriptions = model([
                                                        {"role": "user", "content": story_prompt},
                                                        {"role": "assistant", "content": story},
                                                        {"role": "user", "content": character_prompt},
                                                        {"role": "assistant", "content": character_discriptions},
                                                        {"role": "user", "content": tileset_prompt},
                                                        {"role": "assistant", "content": tileset_discriptions},
                                                        {"role": "user", "content": tileset_map_prompt},
                                                        {"role": "assistant", "content": tileset_map_discriptions},
                                                        {"role": "user", "content": goal_prompt},
                                                        {"role": "assistant", "content": goal_discriptions},
                                                        {"role": "user", "content": object_tile_prompt}
                                                       ], 
                                             max_new_tokens=1500,
                                             temperature = 1)[0]["generated_text"][-1]["content"]
        
        input_prompt = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions + "\n" + tileset_prompt + "\n" + tileset_discriptions + "\n" + tileset_map_prompt + "\n" + tileset_map_discriptions + "\n" + goal_prompt + "\n" + goal_discriptions + "\n" + object_tile_prompt
        input_tokens = self.getTokensSize(input_prompt)
        output_tokens = self.getTokensSize(object_tile_discriptions)
        self.total_input_tokens.append(input_tokens)
        self.total_output_tokens.append(output_tokens)
        print("\n")
        print(object_tile_discriptions)
        object_tile_discriptions = change_answer(object_tile_discriptions)
        object_tiles_list = extract_list(object_tile_discriptions)
        print(object_tiles_list)
        print("\n")

        return object_tiles_list, object_tile_discriptions, object_tile_discriptions

    def world_generation(self, rounds, previous_story, story_paragraphs, total_objectives, previous_tile_map, previous_map, previous_eval, story, story_prompt, character_discriptions, character_discriptions_dict, character_prompt, tileset_discriptions, tileset_prompt, tile_map_dict, tileset_map_discriptions, tileset_map_prompt, goal_discriptions, goal_prompt, important_tiles_list, important_tile_discriptions, important_tile_prompt, walkable_tiles_list, walkable_tile_discriptions, walkable_tile_prompt, save_dir):
        print(f"Generating World...")
        NO_OF_EXCEPTIONS_2 = 0
        no_of_important_tiles = 15
        history_to_keep = 0
        good_feedback_prompt = ""
        bad_feedback_prompt = ""
        good_feedback_check = 0
        bad_feedback_check = 0
        if rounds == 0 or history_to_keep > 0:
            world_system_prompt = "You are a 2D game designer that is profficient in designing tile-based maps. Designing any size of the tile-based map is not a problem for you. This is your first round of generation. You are given the goals to achieve and a list of important tiles to place. Consider them to make the world. Do not place the protagonist, the antagonist and the interactive objects of the story right now. Only create the world right now. Also, consider goals that you extracted earlier and generate while keeping them in context."    
            world_prompt = f"Using the following tile-to-char mapping: {tile_map_dict}, create an entire world on a tile-based grid. \nEnsure that each tile represents a single object, meaning no structures like houses or buildings that require more than one tile.\n Include the following important tiles: {important_tiles_list} and walkable tiles: {walkable_tiles_list}. \n Use exactly {no_of_important_tiles} important tiles to create the world.\n Do not place the protagonist, antagonist, or important objects of the story at this stage. \n Return the world as a string format enclosed with three backticks (```), without any additional text, explanation, or code. If the output contains any strings not part of the mapping, remove them."
        else:
        
            if len(previous_map) > history_to_keep:
                previous_map = previous_map[-history_to_keep:]
                previous_story = previous_story[-history_to_keep:]
                previous_tile_map = previous_tile_map[-history_to_keep:]
                previous_eval = previous_eval[-history_to_keep:]
            history_intro = f"For your reference, here are the previous {len(previous_map)} stories, their tile mapping and corresponding 2D world maps\n"
            for i in range(len(previous_map)):
                history = history_intro + f"Story {i}: {previous_story[i]}\nTile Map for story {i}:\n{previous_tile_map[i]}\n, 2D World map for story {i}:\n {previous_map[i]} and evaluation for the 2D World map:\n{previous_eval[i]}. {good_feedback_prompt}\n{bad_feedback_prompt} Create higher quality and with a higher diversity map."
            world_system_prompt = f"You are a 2D game designer that is profficient in designing tile-based maps. Designing any size of the tile-based map is not a problem for you. This is the generation number {round} for you and you will be provided by previous generation results. Improve evaluation scores in each generation. Previous evaluation scores will be provided to you. You are given the goals to achieve and a list of important tiles to place. Additionally you are given 2D tile-maps and stories that were create before for you to make a better map. Consider them to make the world. Do not place the protagonist, the antagonist and the important objects of the story right now. Only create the world right now. Also, consider goals that you extracted earlier and generate while keeping them in context."    
            world_prompt = f"Using the following tile to character mapping:\n{tile_map_dict}\nCreate an entire world on a tile-based grid. Do not create things that would need more than one tile. For example, a house or a building needs more than one tile to be made. Also, following characters are important to place:\n{important_tiles_list}\n and walkable tiles:\n{walkable_tiles_list}\n Use {no_of_important_tiles} important tiles to create the world. Do not place the protagonist, the antagonist and the important objects of the story right now. Only create the world right now. Create it is a string format with three backticks to start and end with (```) and not in a list format.  {history} \n dont write anything but the string format in the backticks dont write addition text and dont write code no code also dont explain what you did and why."
        done = False
        while not done:
            try:
                world_discriptions  = model([
                                                                                        {"role": "user", "content": story_prompt},
                                                                                        {"role": "assistant", "content": story},
                                                                                        {"role": "user", "content": character_prompt},
                                                                                        {"role": "assistant", "content": character_discriptions},
                                                                                        {"role": "user", "content": tileset_prompt},
                                                                                        {"role": "assistant", "content": tileset_discriptions},
                                                                                        {"role": "user", "content": tileset_map_prompt},
                                                                                        {"role": "assistant", "content": tileset_map_discriptions},
                                                                                        {"role": "user", "content": goal_prompt},
                                                                                        {"role": "assistant", "content": goal_discriptions},
                                                                                        {"role": "user", "content": important_tile_prompt},
                                                                                        {"role": "assistant", "content": important_tile_discriptions},
                                                                                        {"role": "system", "content": world_system_prompt},
                                                                                        {"role": "user", "content": world_prompt}
                                                                                        ], max_new_tokens=1500,
                                                                                        temperature = 1)[0]["generated_text"][-1]["content"]

                print("World: \n")
                print(world_discriptions)
                print("\n")
                input_prompt = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions + "\n" + tileset_prompt + "\n" + tileset_discriptions + "\n" +  tileset_map_prompt + "\n" + tileset_map_discriptions + "\n" + goal_prompt + "\n" + goal_discriptions + "\n" + important_tile_prompt + "\n" + important_tile_discriptions + "\n" + world_system_prompt + "\n" + world_prompt
                self.total_input_tokens.append(self.getTokensSize(input_prompt))
                self.total_output_tokens.append(self.getTokensSize(world_discriptions))
                print("Extracting tilemap..")
                print("\n")
                world_discriptions = change_answer(world_discriptions)
                world_map_raw = extract_between_ticks(world_discriptions)
                print(world_map_raw)
                print("\n")
                print("Fixing tilemap..")
                world_map_raw = world_map_raw.replace(' ', '').replace('"', '')
                world_map_fixed = remove_extra_special_chars(world_map_raw)
            
                print(world_map_fixed)

                used_char_dict = extract_present_elements(world_map_fixed, tile_map_dict)

                world_with_characters_prompt = f"Now that you have created the following world map:\n{world_map_fixed}\n Place only the protagonist, the antagonist and the interactive objects of the story. Do not change anything in the world, just place only the protagonist, the antagonist and the interactive objects in the world."

                world_with_characters_discriptions = model([
                                                                                        {"role": "user", "content": story_prompt},
                                                                                        {"role": "assistant", "content": story},
                                                                                        {"role": "user", "content": character_prompt},
                                                                                        {"role": "assistant", "content": character_discriptions},
                                                                                        {"role": "user", "content": tileset_prompt},
                                                                                        {"role": "assistant", "content": tileset_discriptions},
                                                                                        {"role": "user", "content": tileset_map_prompt},
                                                                                        {"role": "assistant", "content": tileset_map_discriptions},
                                                                                        {"role": "user", "content": goal_prompt},
                                                                                        {"role": "assistant", "content": goal_discriptions},
                                                                                        {"role": "user", "content": important_tile_prompt},
                                                                                        {"role": "assistant", "content": important_tile_discriptions},
                                                                                        {"role": "system", "content": world_system_prompt},
                                                                                        {"role": "user", "content": world_prompt},
                                                                                        {"role": "assistant", "content": world_discriptions},
                                                                                        {"role": "user", "content": world_with_characters_prompt},
                                                                                        ], max_new_tokens=1500,
                                                                                        temperature = 1)[0]["generated_text"][-1]["content"]

                input_prompt_wwcd = story_prompt + "\n" + story + "\n" + character_prompt + "\n" + character_discriptions + "\n" + tileset_prompt + "\n" + tileset_discriptions + "\n" +  tileset_map_prompt + "\n" + tileset_map_discriptions + "\n" + goal_prompt + "\n" + goal_discriptions + "\n" + important_tile_prompt + "\n" + important_tile_discriptions + "\n" + world_system_prompt + "\n" + world_prompt + "\n" + world_discriptions + "\n" + world_with_characters_prompt

                print("World: \n")
                print(world_with_characters_discriptions)
                print("\n")
                self.total_input_tokens.append(self.getTokensSize(input_prompt_wwcd))
                self.total_output_tokens.append(self.getTokensSize(world_with_characters_discriptions))
                print("Extracting tilemap..")
                print("\n")
                world_with_characters_discriptions = change_answer(world_with_characters_discriptions)
                world_map_raw_with_chars = extract_between_ticks(world_with_characters_discriptions)
                print(world_map_raw_with_chars)
                print("\n")
                print("Fixing tilemap..")
                world_map_raw_with_chars = world_map_raw_with_chars.replace(' ', '').replace('"', '')
                world_map_fixed_with_chars = remove_extra_special_chars(world_map_raw_with_chars)
                print(world_map_fixed_with_chars)
            
                used_char_dict_with_char = extract_present_elements(world_map_fixed_with_chars, tile_map_dict)

                color_code_prompt = f"Take the following Python dictionary:\n{used_char_dict_with_char}\n and create another dictionary that has Keys as the values of the dictionary above and values as appropriate hexadecimal color codes. For example, grass will have green hexadecimal color code. Stricty return only a Python dictionary. Do not return it in a Python response and do not return any text other then the dictionary"

                color_code_discriptions = model([
                                                                                        {"role": "user", "content": color_code_prompt},
                                                                                        ], max_new_tokens=1500,
                                                                                        temperature = 1)[0]["generated_text"][-1]["content"]
                print("color_code_discriptions:\n")
                print(color_code_discriptions)
                color_code_discriptions = change_answer(color_code_discriptions)
                char_color_map_with_char_dict = extract_dict(color_code_discriptions)
                char_color_map_with_char = get_image_color_tile_mapping(char_color_map_with_char_dict)

                print(f"char_color_map: {char_color_map_with_char_dict}")
                #colored_tilemap_img_with_char = create_colored_tilemap_image(world_map_fixed_with_chars, char_color_map_with_char_dict)
                #plt.imshow(colored_tilemap_img_with_char)
                #plt.axis('off')
                #plt.savefig(save_dir + f'/world_color_map_with_chars_{rounds}.png', format='png', dpi=150, bbox_inches='tight')
                #plt.show()
            
                world_legend = create_legend_image(char_color_map_with_char_dict, used_char_dict_with_char)
                world_legend.savefig(save_dir + f'/world_color_legend_with_chars_{rounds}.png', format='png', dpi=150, bbox_inches='tight')
                #plt.show()


                evaluator = LlamaEvaluator(self.total_input_tokens, self.total_output_tokens)
                world_eval_dict, self.total_input_tokens, self.total_output_tokens = evaluator.evaluate_world(map=world_map_fixed_with_chars,
                                                                                                                tile_map_dictionary=used_char_dict_with_char,
                                                                                                                story=story,
                                                                                                                walkable_tiles=walkable_tiles_list,
                                                                                                                important_tiles=important_tiles_list,
                                                                                                                previous_maps=previous_map)
            

                llm_agent_reward, astar_path, objectives = self.action_generation(rounds,story,"protagonist","antagonist", character_discriptions_dict,world_map_fixed,world_map_fixed_with_chars,used_char_dict,used_char_dict_with_char,"color_tiles_img_with_char",
                        "char_color_map",walkable_tile_discriptions,important_tile_discriptions,goal_discriptions, save_dir)
            
                #agent_reward = -1000
                world_eval_dict["agent_reward"] = llm_agent_reward
                world_eval_dict["astar_path"] = astar_path


                story_paragraphs, total_objectives, no_of_important_tiles, bad_feedback_prompt, good_feedback_prompt = self.feedback_checks(rounds, world_eval_dict, previous_eval, story_paragraphs, total_objectives, no_of_important_tiles)
                
                done = True

            except Exception as e:
            
                tb = traceback.format_exc()
                print(f"Exception raised: {e}\n {tb}")
                NO_OF_EXCEPTIONS_2 += 1

                if NO_OF_EXCEPTIONS_2 >= 2:
                    done = True
                pass

        color_tiles_img_with_char = ""

        return world_map_fixed, world_map_fixed_with_chars, world_eval_dict, used_char_dict, used_char_dict_with_char, char_color_map_with_char, \
                color_tiles_img_with_char, story_paragraphs, objectives, total_objectives, good_feedback_check, bad_feedback_check, no_of_important_tiles, llm_agent_reward, astar_path

    def action_generation(self, round,
                        story,
                        protagonist_name,
                        antagonist_name,
                        character_discriptions_dict,
                        world_map_fixed,
                        world_map_fixed_with_chars,
                        tileset_used_dict_1st_layer,
                        tileset_used_dict,
                        color_tiles_img_with_char,
                        char_color_map,
                        walkable_tiles_list,
                        object_tiles_list,
                        goal_discriptions, 
                        save_dir):

        
        print("Generating Actions...")
        except_done = False
        NO_OF_EXCEPTIONS_3 = 0
        total_reward = 0
        frames = [] 
        total_episodes = 1
        episodes = 0
        all_episodes_rewards = []
        try:
            while not except_done:
                
                
                objective_tile_system = "You are a great planner in 2D game. You plan objectives for the protagonist of the game. All objectives should match the goals extracted from the story. Objectives should strictly follow them. Return a Python dictionary of the objective as the key and a tile that achieves the objective and the position of the tile. For example 'Objective': ['A', 6, 1]. Only return a Python dictionary. Do not return a python response."
                objective_tile_prompt = f"Given the story:\n{story}\n a 2D tile map of a world was created for the story:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping:\n{tileset_used_dict}\n You are also provided with the description of the goals:\n{goal_discriptions}\n and walkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n Taking this information into your context, create the objectives to achieve and also provide the tile that you will pick up, reach the position at or hit the enemy. Return a Python dictionary of the objective as the key and a tile that achieves the objective and the position of the tile. The pattearn should be 'Objective': ['tile', row, column], for example 'Objective': ['A', 6, 1], thus the first element would be the tile and second and third elements of the list will be position of the tile. Return strictly in this format.  dont write anything but the format! dont write addition text! and dont write a python code!  dont explain what you did and why!"
                
                objective_tile_discriptions = model([
                                                                    {"role": "system", "content": objective_tile_system},
                                                                    {"role": "user", "content": objective_tile_prompt}
                                                        ],
                                                        max_new_tokens=1500)[0]["generated_text"][-1]["content"]
        
                input_prompt = objective_tile_system + "\n" + objective_tile_prompt
                input_tokens = self.getTokensSize(input_prompt)
                output_tokens = self.getTokensSize(objective_tile_discriptions)
                self.total_input_tokens.append(input_tokens)
                self.total_output_tokens.append(output_tokens)
                
                
                

                print("Objectives: \n")
                print(objective_tile_discriptions)
                print("\n")
                objective_tile_discriptions = change_answer(objective_tile_discriptions)
                objective_tile_dict = extract_dict(objective_tile_discriptions)

                print("objective_tile in a dict: \n")
                print(objective_tile_dict)
                print("\n")

                total_actions = {}
                objective_flag = False

                walkable_tiles_list = extract_list(walkable_tiles_list)

                print("walkable_tiles_list: \n")
                print(walkable_tiles_list)
                print("\n")
                object_tiles_list = extract_list(object_tiles_list)
                print("object_tiles_list: \n")
                print(object_tiles_list)
                print("\n")
                # ASTAR Search
                world_map_fixed_with_chars = pad_rows_to_max_length(world_map_fixed_with_chars)
                parsed_world_map = parse_grid(world_map_fixed_with_chars)

                objective_tile_list = []
                for _keys, str_obj in objective_tile_dict.items():
                    temp_list = extract_list(str(str_obj))
                    objective_tile_list.append((temp_list[1], temp_list[2]))
                print("objective_tile_list: \n")
                print(objective_tile_list)
                print("\n")
                solving = False
                solving_exceptions = 0
                while not solving:
                    try:
                        # Initialize the game state and agent
                        game_state = WorldState(walkable_tiles_list, object_tiles_list, parsed_world_map, objective_tile_list)
                        game_state = game_state.stringInitialize(parsed_world_map, objective_tile_list)
                        astar_agent = EnhancedAStarWorldAgent(walkable_tiles_list, objective_tile_list, game_state, object_tiles_list)
                        astar_path, _, _, game_map_updated, _ = astar_agent.getSolution(game_state,maxIterations=10000)
                        print(f"astar_path: {len(astar_path)}")
                        solving = True
                    except Exception as e:
                        #print(f"check#3 done = {done}")
                        tb = traceback.format_exc()
                        print(f"Exception raised: {e}\n {tb}")
                        solving_exceptions += 1
                        if solving_exceptions >= 5:
                            solving = True
                            astar_path = []
                            print(f"astar_path: {len(astar_path)}")
                        pass
                    

                removed_value = tileset_used_dict.pop('Protagonist', None) # TODO - CHECK WHAT THIS PART DOES AND FIX
                removed_value = tileset_used_dict.pop('Antagonist', None) 
                tileset_used_dict[character_discriptions_dict["Protagonist"]] = "@"
                tileset_used_dict[character_discriptions_dict["Antagonist"]] = "#"
                print("Retrieving images.\n")
                tile_images,_s= find_most_similar_images(tileset_used_dict,cfg.tile_data_dir)
                print("Images Retrieved.")
                tile_images_1st_layer = overlap_dict(tile_images, tileset_used_dict_1st_layer)
                print("after overlap_dict\n")
                legend = create_legend(tile_images,tileset_used_dict)
                print("after create legen\n")
                #plt.imshow(legend)
                #plt.axis('off')
                #plt.savefig(save_dir + f'/world_legend_with_chars_{round}.png', format='png', dpi=150, bbox_inches='tight')
                #plt.show()
                legend.save(save_dir + f'/world_legend_with_chars_{round}.png', 'PNG')
                print("after legend save\n")

                env = Word2WorldEnv(walkable_tiles_list, tile_images_1st_layer, tile_images, world_map_fixed, world_map_fixed_with_chars, object_tiles_list, "#")
                print("after Word2WorldEnv\n")
                agent = LLMAgent()
                print("after LLMAgent\n")
                state = env.reset()
                print("env.reset\n")
                env_image = env.render(mode="image")
                print("after env.render\n")
                env_image.save(save_dir + f'/world_map_with_chars_{round}.png', 'PNG')
                print("after env_image.save\n")

                reward_feedback = "This is your first objective"
                reward_design = {
                    "You are 8 tiles away from objective thus objective is incomplete": -100,
                    "You are 5 to 8 tiles away from objective thus objective is incomplete": -50,
                    "You are 3 to 5 tiles away from objective": +25,
                    "You are 1 to 3 tiles away from objective": +50,
                    "You are 1 tile away or your are on the objective tile from objective": +100,
                    "You have completed the objective": +100,
                }
                protagonist_position = find_character_position(world_map_fixed_with_chars, "@")
                print("after find_character_position\n")
                prev_episode_reward = 0
                done = False
                
                while not done:
                    reward = 0
                    for i in range(len(objective_tile_dict)):
                        
                        print("\n")
                        print(f"OBJECTIVE: {list(objective_tile_dict.keys())[i]}")
                        print("\n")
                        total_actions[list(objective_tile_dict.keys())[i]] = []
                        #while not objective_flag:
                        action_system = f"You are a great planner in 2D game. You plan actions for the protagonist of the game to achieve all objects. You are given objectives, tiles and the position of tiles to achieve the objectives. You have the following options as actions: 'move_up', move_down, 'move_right', 'move_left', 'pick_object', 'hit_enemy'. Generate a sequence of actions that will achieve the objective. Only return the sequence of actions from the options."
                        action_prompt = f"Given the story:\n{story}\n a 2D tile map of a world was created for the story:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\n and walkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Accumulative rewards for all the previous objectives tille now are {reward}. Taking this information into your context, create a sequence of actions for the protagonist to complete the objective: {list(objective_tile_dict.keys())[i]}, which is to reach the tile, 'pick_object' or 'hit_enemy' at tile and position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary. Do not return it in a Python response.  Return strictly in this format.  dont write anything but the format! dont write addition text! and dont write a python code!  dont explain what you did and why!"
                        
                        actions_discriptions = model([
                                                                    {"role": "system", "content": action_system},
                                                                    {"role": "user", "content": action_prompt}
                                                                   ],
                                                                   max_new_tokens=1500)[0]["generated_text"][-1]["content"]
        
                        input_prompt = action_system + "\n" + action_prompt
                        input_tokens = self.getTokensSize(input_prompt)
                        output_tokens = self.getTokensSize(actions_discriptions)
                        self.total_input_tokens.append(input_tokens)
                        self.total_output_tokens.append(output_tokens)
                        print("actions_discriptions:\n")
                        print(actions_discriptions)
                        print("\n")
                        actions_discriptions = change_answer(actions_discriptions)
                        action_dict = extract_dict(actions_discriptions)
                        print("Action: \n")
                        if "action" not in action_dict.keys():
                            action_dict["action"] = ""
                        print(action_dict["action"])
                        print("\n")
                        total_actions[list(objective_tile_dict.keys())[i]].append(action_dict["action"])
                        
                        
                        for action_str in action_dict["action"]:
                            action = agent.action(action_str)
                            state, _r, done, _ = env.step(action)
                            
                            frame = env.render(mode='rgb_array')  # Capture the frame
                            frames.append(frame)  # Append the frame
                            time.sleep(0.01)
                    
                    
                        current_state = list_of_lists_to_string(state)
                        
                        print(current_state)
                        print("\n")
                        
                        check_prompt = f"Given the previous world state:\n{world_map_fixed_with_chars}\n and the updated state that you returned: \n{current_state}\n is the objective {list(objective_tile_dict.keys())[i]} completed? Remember, from the dictionary of objectives, this objective will be completed when you reach tile {list(objective_tile_dict.values())[0]} at position {list(objective_tile_dict.values())[1]} or you are one tile aound this position in any directions. Strictly, only return 'Complete' or 'Incomplete'."
                        
                        check_discriptions = model([
                                                                    {"role": "system", "content": action_system},
                                                                    {"role": "user", "content": action_prompt},
                                                                    {"role": "assistant", "content": actions_discriptions},
                                                                    {"role": "user", "content": check_prompt}
                                                                 ],
                                                                 max_new_tokens=1500)[0]["generated_text"][-1]["content"]
        
                        input_prompt = action_system + "\n" + action_prompt + "\n" + actions_discriptions + "\n" + check_prompt
                        input_tokens = self.getTokensSize(input_prompt)
                        output_tokens = self.getTokensSize(check_discriptions)
                        self.total_input_tokens.append(input_tokens)
                        self.total_output_tokens.append(output_tokens)
                        world_map_fixed_with_chars = current_state
                        
                        
                        for k, value in enumerate(objective_tile_dict.values()):
                            if k == i:
                                objective_pos = extract_list(str(value))
                        protagonist_position = find_character_position(world_map_fixed_with_chars, "@")
                        print("\n")
                        print(f"protagonist_position: {protagonist_position}")
                        print(f"objective_position: [{objective_pos[1]},{objective_pos[2]}]")
                        

                        distance_from_objective = (abs(objective_pos[1] - protagonist_position[0]), abs(objective_pos[2] - protagonist_position[1]))
                        print(f"distance from current objective: [{distance_from_objective[0]}, {distance_from_objective[1]}]") 
                        print("\n")
                        reward_feedback = ""
                        reward_feedback = "Your previous objectives reward feedback is: "
                        if (distance_from_objective[0] > 8 or distance_from_objective[1] > 8):
                            reward -= 100
                            reward_feedback += f"You were very far from the objective tile so you were given a regret(negative reward) of -100 points and objective was INCOMPLETE"
                        if (distance_from_objective[0] > 5 and distance_from_objective[0] < 8) or (distance_from_objective[1] > 5 and distance_from_objective[1] < 8):
                            reward -= 50
                            reward_feedback += f"You were far from the objective tile so you were given a regret(negative reward) of -50 points and objective was INCOMPLETE"
                        if (distance_from_objective[0] <= 5 and distance_from_objective[0] > 3) and (distance_from_objective[1] <= 5 and distance_from_objective[1] > 3):
                            reward += 25
                            reward_feedback += f"You were close to the objective tile so you were given a reward of 25 points"
                        if (distance_from_objective[0] < 3 and distance_from_objective[0] > 1) and (distance_from_objective[1] < 3 and distance_from_objective[1] > 1):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"

                        if (distance_from_objective[0] <= 1) and (distance_from_objective[1] > 1 and distance_from_objective[1] <= 5):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"
                        if (distance_from_objective[1] <= 1) and (distance_from_objective[0] > 1 and distance_from_objective[0] <= 5):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"

                        if (distance_from_objective[0] <= 1 and distance_from_objective[1] <= 1) or check_discriptions == "Complete":
                            
                            if (distance_from_objective[0] == 0 and distance_from_objective[1] == 0) and check_discriptions == "Complete":
                                reward += 200
                                #objective_flag = True
                                reward_feedback += f"You were by the objective tile and you COMPLETED the objective so you were given a reward of 200 points"
                            else:
                                reward += 100
                                reward_feedback += f"You were by the objective tile so you were given a reward of 100 points"
                        print("\n")
                        print(f"EPISODE REWARDS uptill now: {reward}")
                        print("\n")
                    total_reward += reward
                    episodes += 1
                    all_episodes_rewards.append(reward)
                    print("\n")
                    print(f"TOTAL REWARD for EPISODE: {total_reward}")
                    if episodes == total_episodes:
                        done = True
                
                    with imageio.get_writer(f'{cfg.save_dir}_{round}.mp4', fps=10) as video:
                        for frame in frames:
                            video.append_data(frame)

                except_done = True
            
        except Exception as e:
            #print(f"check#3 done = {done}")
            tb = traceback.format_exc()
            print(f"Exception raised: {e}\n {tb}")
            NO_OF_EXCEPTIONS_3 += 1
            if NO_OF_EXCEPTIONS_3 >= 5:
                except_done = True
            pass


        return max(all_episodes_rewards), len(astar_path), objective_tile_dict
    
