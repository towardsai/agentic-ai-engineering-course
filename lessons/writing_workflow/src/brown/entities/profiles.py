from pydantic import BaseModel

from brown.entities.mixins import ContextMixin


class Profile(BaseModel, ContextMixin):
    name: str
    content: str

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    <name>{self.name}</name>
    <content>{self.content}</content>
</{self.xml_tag}>
"""


class CharacterProfile(Profile):
    pass


class ArticleProfile(Profile):
    pass


class StructureProfile(Profile):
    pass


class MechanicsProfile(Profile):
    pass


class TerminologyProfile(Profile):
    pass


class TonalityProfile(Profile):
    pass


class ArticleProfiles(BaseModel):
    character: CharacterProfile

    article: ArticleProfile

    structure: StructureProfile
    mechanics: MechanicsProfile
    terminology: TerminologyProfile
    tonality: TonalityProfile
