import random
from typing import Union, Dict, Any

import nlpaug.augmenter.word as naw
from datasets import Dataset
from datasets import load_dataset
from transformers import BertTokenizer


def get_snli_dataset() -> Union:
    print("Getting the SNLI datasets")
    train = load_dataset('snli', split='train')
    train = train.map(lambda example: {'text': example['premise'] + example['hypothesis']})
    train = train.remove_columns(["premise", "hypothesis"])
    train = train.map(lambda example: {'label': max(0, int(example['label']))})
    train = train.rename_column("label", "labels")

    valid_data = load_dataset('snli', split='validation')
    valid_data = valid_data.map(lambda example: {'text': example['premise'] + example['hypothesis']})
    valid_data = valid_data.map(lambda example: {'label': max(0, int(example['label']))})
    valid_data = valid_data.remove_columns(["premise", "hypothesis"])
    valid_data = valid_data.rename_column("label", "labels")
    print("Finished getting the  SNLI datasets")

    return train, valid_data


def get_million_headlines() -> Dataset:
    dataset = load_dataset("DeveloperOats/Million_News_Headlines")
    dataset = dataset.rename_column("headline_text", "text")
    dataset = dataset.remove_columns("publish_date")
    dataset = dataset["train"]

    return dataset


def get_pedia_classes() -> Dataset:
    dataset = load_dataset("DeveloperOats/DBPedia_Classes")
    dataset = dataset.remove_columns(["l1", "l2"])
    dataset = dataset["train"]

    labels = ['Senator', 'Album', 'Mountain', 'AcademicJournal', 'HistoricBuilding', 'Reptile', 'MilitaryUnit', 'Judge',
              'ChessPlayer', 'TradeUnion', 'Musical', 'Insect', 'Town', 'SupremeCourtOfTheUnitedStatesCase',
              'TennisPlayer', 'SpeedwayRider', 'Publisher', 'GolfPlayer', 'HorseRider', 'ShoppingMall', 'Road',
              'MilitaryPerson', 'Election', 'ArtistDiscography', 'OfficeHolder', 'IceHockeyPlayer', 'Diocese',
              'Library', 'Lake', 'Monarch', 'Conifer', 'Bridge', 'Dam', 'BasketballPlayer', 'Glacier',
              'NationalFootballLeagueSeason', 'RugbyPlayer', 'Single', 'Mayor', 'VoiceActor', 'SoccerManager',
              'Economist', 'Skier', 'HandballPlayer', 'Saint', 'RecordLabel', 'AmateurBoxer', 'HollywoodCartoon',
              'Theatre', 'Legislature', 'ComicsCreator', 'GolfTournament', 'MilitaryConflict', 'CollegeCoach',
              'AnimangaCharacter', 'OlympicEvent', 'Convention', 'University', 'BadmintonPlayer', 'NCAATeamSeason',
              'VideoGame', 'Gymnast', 'Amphibian', 'Band', 'PublicTransitSystem', 'Bird', 'GaelicGamesPlayer',
              'Volcano', 'Hotel', 'Architect', 'MartialArtist', 'BusCompany', 'RadioStation', 'Crustacean', 'Airport',
              'Lighthouse', 'Castle', 'SoccerPlayer', 'BaseballPlayer', 'HorseRace', 'Swimmer', 'Congressman',
              'Stadium', 'Painter', 'MemberOfParliament', 'ChristianBishop', 'CyclingRace', 'RugbyClub', 'HorseTrainer',
              'Engineer', 'Fish', 'Arachnid', 'Prison', 'GrandPrix', 'Fungus', 'MountainRange', 'BroadcastNetwork',
              'PoliticalParty', 'Entomologist', 'Planet', 'AutomobileEngine', 'Moss', 'BaseballSeason', 'Baronet',
              'ArtificialSatellite', 'Manga', 'Play', 'HockeyTeam', 'SoccerTournament', 'SoapCharacter', 'Cycad',
              'Cyclist', 'Medician', 'President', 'School', 'TelevisionStation', 'BasketballTeam', 'Mollusca',
              'BeautyQueen', 'Noble', 'SoccerClubSeason', 'CyclingTeam', 'FootballMatch', 'PrimeMinister',
              'MusicFestival', 'Philosopher', 'RadioHost', 'Hospital', 'RaceHorse', 'Museum', 'Earthquake', 'Poem',
              'Anime', 'RollerCoaster', 'AdultActor', 'Ambassador', 'Bank', 'FigureSkater', 'RailwayLine',
              'FashionDesigner', 'Newspaper', 'Airline', 'Historian', 'MythologicalFigure',
              'EurovisionSongContestEntry', 'WrestlingEvent', 'Journalist', 'Racecourse', 'BasketballLeague',
              'SoccerLeague', 'Religious', 'Village', 'Jockey', 'BusinessPerson', 'MusicGenre', 'HandballTeam',
              'AmericanFootballPlayer', 'BiologicalDatabase', 'Canoeist', 'FormulaOneRacer', 'SumoWrestler',
              'TennisTournament', 'River', 'Grape', 'Skater', 'RugbyLeague', 'ClassicalMusicComposition', 'Fern',
              'NascarDriver', 'Restaurant', 'Cricketer', 'Bodybuilder', 'Chef', 'CultivatedVariety', 'Governor',
              'Cardinal', 'SolarEclipse', 'TableTennisPlayer', 'Model', 'FilmFestival', 'ScreenWriter', 'Brewery',
              'SquashPlayer', 'AustralianRulesFootballPlayer', 'Magazine', 'GolfCourse',
              'WomensTennisAssociationTournament', 'LawFirm', 'Comedian', 'MountainPass', 'Winery', 'GreenAlga',
              'RailwayStation', 'PlayboyPlaymate', 'Galaxy', 'RoadTunnel', 'Rower', 'SportsTeamMember',
              'MixedMartialArtsEvent', 'Photographer', 'CanadianFootballTeam', 'DartsPlayer', 'CricketTeam',
              'ClassicalMusicArtist', 'Cave', 'CricketGround', 'AustralianFootballTeam', 'PokerPlayer', 'Poet',
              'Curler', 'Astronaut', 'BaseballLeague', 'Pope', 'BeachVolleyballPlayer', 'ComicStrip', 'NetballPlayer',
              'IceHockeyLeague', 'Canal', 'LacrossePlayer']

    dataset = dataset.map(lambda example: {'labels': labels.index(example["l3"])})
    dataset = dataset.remove_columns("l3")

    return dataset


def preprocess_datasets(tokenizer: BertTokenizer, data_set: Dataset) -> Dataset:
    print("Preprocess the data")
    data_set = data_set.map(augment_dataset)

    data_set = data_set.map(
        lambda data_point: tokenizer(data_point['text'], padding=True, truncation=True),
        batched=True)

    if 'labels' in data_set:
        data_set.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    else:
        data_set.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask'])

    print("Finsihed the Preprocess the data")
    return data_set


def augment_dataset(text: str) -> Dict[str, Any]:
    tuples = [
        None,
        naw.SynonymAug(aug_src='wordnet'),
        naw.ContextualWordEmbsAug(
            model_path='distilbert-base-uncased', action="substitute"),
        naw.RandomWordAug(action='crop'),
        naw.RandomWordAug(),
        naw.ContextualWordEmbsAug(
            model_path='roberta-base', action="substitute"),
        naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="insert"),
    ]
    aug = random.choices(tuples, weights=(50, 10, 10, 10, 10, 10, 10), k=1)[0]
    # back_translation_aug = naw.BackTranslationAug(
    #     from_model_name='facebook/wmt19-en-de',
    #     to_model_name='facebook/wmt19-de-en'
    # )
    # back_translation_aug.augment(str(text))
    if aug is None:
        return text
    else:
        augmented_text = aug.augment(str(text))
        return {"data": augmented_text}
