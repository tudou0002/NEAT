
class Entity:
    def __init__(self, text: str, category:str, begin_offset: int):
        """Construct an entity object.

        text: the plain text of this entity.
        category: The type of entity
        begin_offset: The index of the entity within the parent document.

        """
        self.text = text
        self.category = category
        self.begin_offset = begin_offset
        self.end_offset = begin_offset + len(text)

    def __len__(self):
        """The number of unicode characters in the entity, i.e. `entity.text`.

        RETURNS (int): The number of unicode characters in the entity.
        """
        return self.text.length

    def __eq__(self, other):
        """Two entities will equal to each other if they have the same class, 
        same attributes and from the same parent document. 

        """
        return self.text == other.text and self.category == other.category and self.begin_offset == other.begin_offset

    def __hash__(self):
        return hash((self.text, self.category, self.begin_offset))
