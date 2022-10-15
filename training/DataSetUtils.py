import logging
import random
from typing import Union, Dict, Any

import nlpaug.augmenter.word as naw
from datasets import load_dataset, Dataset
from transformers import BertTokenizer

logging.disable(logging.INFO)  # disable INFO and DEBUG logger everywhere
tuples = [
    None,
    naw.SynonymAug(aug_src='wordnet', aug_max=2),
    naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased', aug_max=2, action="substitute", device="cuda"),
    # naw.RandomWordAug(action='crop'),
    # naw.RandomWordAug(),
    # naw.ContextualWordEmbsAug(
    #     model_path='roberta-base', action="substitute", device="cuda"),
    naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', aug_max=2, action="insert", device="cuda"),
]


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
    dataset = dataset.select(range(1, 100000))

    return dataset


def get_tec() -> Dataset:
    dataset = load_dataset("trec")
    dataset = dataset["train"]
    dataset = dataset.rename_column("label-fine", "original_label")

    return dataset


def get_imdb() -> Dataset:
    dataset = load_dataset("imdb")
    dataset = dataset["train"]
    dataset = dataset.rename_column("label", "original_label")

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


def preprocess_datasets(tokenizer: BertTokenizer, new_dataset: Dataset) -> Dataset:
    print("Preprocess the data")
    new_dataset = new_dataset.map(augment_dataset, batch_size=2, batched=True, load_from_cache_file=False)

    new_dataset = new_dataset.map(
        lambda data_point: tokenizer(data_point['text'], padding=True, truncation=True),
        batched=True)

    if 'labels' in new_dataset:
        new_dataset.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    else:
        new_dataset.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask'])

    print("Finished the Preprocess the data")
    return new_dataset


def augment_dataset(data_point) -> Dict[str, Any]:
    aug = random.choices(tuples, weights=(100, 10, 10, 10), k=1)[0]

    if aug is None:
        return data_point
    else:
        texts = [str(i) for i in data_point.data["text"] if i]
        augmented_text = aug.augment(texts)
        data_point["text"] = augmented_text
        return data_point
