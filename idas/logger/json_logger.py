#  Copyright 2019 Gabriele Valvano
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import io

data = {}


def add_new_node(key, values, file_name='log_file.json'):
    """
    Add new node to JSON file under the key=key and with sub-keys=values.

    Args:
        key (string): dictionary key to address the node
        values (dict): dictionary with key-values couples
        file_name (string): JSON file name to write

    Example:
        data.update({'SPARSE_TRAINING': {'done_before': False, 'beta': 0.10, 'sparsity': 0.30}})

    """
    data.update({key: values})

    with io.open(file_name, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)


def read_one_node(key, file_name='log_file.json'):
    """
    Return the dictionary in JSON file under the key=key.

    Args:
        key (string): dictionary key to address the node
        file_name (string): JSON file name to read

    Returns:
        Dictionary

    """
    with open(file_name, 'r', encoding='utf8') as infile:
        node = json.load(infile)
    return node[key]


def update_node(key, sub_key, sub_value, file_name='log_file.json'):
    """
    Update a node in a JSON file under the key=key and with sub-keys=values.

    Args:
        key (string): dictionary key to address the node
        sub_key (string): field name to be updated under the node key
        sub_value (): value to assign to the field name under the node key
        file_name (string): JSON file name to write

    """
    content_dict = read_one_node(key, file_name=file_name)
    content_dict[sub_key] = sub_value

    data.update({key: content_dict})

    with io.open(file_name, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)


if __name__ == '__main__':
    # Example of reading and writing on JSON file
    k = 'KEY_1'
    val = {'flag': True, 'alpha': 0.10, 'beta': 0.30}
    add_new_node(k, val)

    k = 'KEY_2'
    val = {'flag': False, 'alpha': 0.20, 'beta': 0.60}
    add_new_node(k, val)

    k = 'KEY_1'
    print(read_one_node(k))
