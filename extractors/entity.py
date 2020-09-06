
class Entity:
    def __int__(self, text: str, category:str, begin_offset: int):
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


