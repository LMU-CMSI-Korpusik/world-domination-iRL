from riskGame import Territory, Continent

# North America:
alaska = Territory('Alaska')
northwest_territory = Territory('Northwest Territory')
alberta = Territory('Alberta')
ontario = Territory('Ontario')
greenland = Territory('Greenland')
quebec = Territory('Quebec')
west_us = Territory('Western U.S.')
east_us = Territory('Eastern U.S.')
central_america = Territory('Central America')

na_territories = frozenset([alaska, northwest_territory, alberta,
                            ontario, greenland, west_us, east_us, central_america])
north_america = Continent('North America', na_territories, 5)

# South America:
venezuela = Territory('Venezuela')
peru = Territory('Peru')
brazil = Territory('Brazil')
argentina = Territory('Argentina')

sa_territories = frozenset([venezuela, peru, brazil, argentina])
south_america = Continent('South America', sa_territories, 2)

# Africa:
north_africa = Territory('North Africa')
egypt = Territory('Egypt')
east_africa = Territory('East Africa')
congo = Territory('Congo')
south_africa = Territory('South Africa')
madagascar = Territory('Madagascar')

africa_territories = frozenset(
    [north_africa, egypt, east_africa, congo, south_africa, madagascar])
africa = Continent('Africa', africa_territories, 3)

# Europe:
iceland = Territory('Iceland')
scandinavia = Territory('Scandinavia')
ukraine = Territory('Ukraine')  # Slava Ukraini
great_britain = Territory('Great Britain')
north_europe = Territory('Northern Europe')
west_europe = Territory('Western Europe')
south_europe = Territory('Southern Europe')

europe_territories = frozenset([iceland, scandinavia, ukraine,
                                great_britain, north_europe, west_europe, south_europe])
europe = Continent('Europe', europe_territories, 5)

# Asia
ural = Territory('Ural')
siberia = Territory('Siberia')
yakutsk = Territory('Yakutsk')
kamchatka = Territory('Kamchatka')
irkutsk = Territory('Irkutsk')
afghanistan = Territory('Afghanistan')
china = Territory('China')
mongolia = Territory('Mongolia')
japan = Territory('Japan')
middle_east = Territory('Middle East')
india = Territory('India')
siam = Territory('Siam')

asia_territories = frozenset([ural, siberia, yakutsk, kamchatka, irkutsk,
                              afghanistan, china, mongolia, japan, middle_east, india, siam])
asia = Continent('Asia', asia_territories, 7)

# Australia
indonesia = Territory('Indonesia')
new_guinea = Territory('New Guinea')
west_au = Territory('Western Australia')
east_au = Territory('Eastern Australia')

au_territories = frozenset([indonesia, new_guinea, west_au, east_au])
australia = Continent('Australia', au_territories, 2)

classic_territories = {
    alaska: {northwest_territory, alberta, kamchatka},
    northwest_territory: {alaska, alberta, ontario, greenland},
    greenland: {northwest_territory, ontario, quebec, iceland},
    alberta: {alaska, northwest_territory, ontario, west_us},
    ontario: {alberta, northwest_territory, greenland, quebec, east_us, west_us},
    quebec: {ontario, greenland, east_us},
    west_us: {alberta, ontario, east_us, central_america},
    east_us: {west_us, ontario, quebec, central_america},
    central_america: {west_us, east_us, venezuela},
    venezuela: {central_america, peru, brazil},
    peru: {venezuela, brazil, argentina},
    brazil: {venezuela, peru, argentina, north_africa},
    argentina: {peru, brazil},
    north_africa: {brazil, west_europe, south_europe, egypt, east_africa, congo},
    egypt: {north_africa, south_europe, middle_east, east_africa},
    east_africa: {egypt, north_africa, congo, madagascar, south_africa},
    congo: {north_africa, east_africa, south_africa},
    south_africa: {congo, east_africa, madagascar},
    madagascar: {east_africa, south_africa},
    iceland: {greenland, great_britain, scandinavia},
    great_britain: {iceland, scandinavia, north_europe, west_europe},
    scandinavia: {iceland, great_britain, north_europe, ukraine},
    north_europe: {great_britain, scandinavia, ukraine, west_europe, south_europe},
    ukraine: {scandinavia, ural, north_europe, south_europe, afghanistan, middle_east},
    west_europe: {great_britain, north_europe, south_europe, north_africa},
    south_europe: {west_europe, north_europe, ukraine, middle_east, north_africa, egypt},
    ural: {ukraine, afghanistan, siberia, china},
    siberia: {ural, china, yakutsk, irkutsk, mongolia},
    yakutsk: {siberia, irkutsk, kamchatka},
    kamchatka: {alaska, yakutsk, irkutsk, mongolia, japan},
    irkutsk: {siberia, yakutsk, kamchatka, mongolia},
    mongolia: {siberia, irkutsk, kamchatka, japan, china},
    afghanistan: {ukraine, ural, china, middle_east, india},
    china: {afghanistan, ural, siberia, mongolia, india, siam},
    middle_east: {south_europe, ukraine, afghanistan, india, egypt},
    india: {middle_east, afghanistan, china, siam},
    siam: {india, china, indonesia},
    indonesia: {siam, new_guinea, west_au},
    new_guinea: {indonesia, west_au, east_au},
    west_au: {indonesia, new_guinea, east_au},
    east_au: {new_guinea, west_au},
    japan: {kamchatka, mongolia}
}

classic_continents = frozenset([north_america,
                                south_america, europe, asia, africa, australia])

for territory in classic_territories.keys():
    for neighbor in classic_territories[territory]:
        try:
            assert territory in classic_territories[neighbor]
        except:
            print(f'{territory.name} not in neighbors of {neighbor.name}')
