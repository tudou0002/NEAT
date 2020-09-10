
class Entity:
    def __init__(self, text: str, begin_offset: int, score=-1.0):
        """Construct an entity object.

        text: the plain text of this entity.
        score: The certainty of this entity being a name. 
            Default is -1.0 meaning the certainty is not calculated by word embedding.
        begin_offset: The index of the entity within the parent document.

        """
        self.text = text
        self.score = score
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
        return self.text == other.text and self.begin_offset == other.begin_offset

    def __hash__(self):
        return hash((self.text, self.begin_offset))
