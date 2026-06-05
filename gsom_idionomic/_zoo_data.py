"""Embedded UCI Zoo dataset (101 animals x 16 traits + class).

Source: UCI Machine Learning Repository, Zoo Data Set (R. Forsyth).
Public domain. Bundled so the sample works with no file/network access.
"""

ZOO_CSV = """ID,hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail,domestic,catsize,class
aardvark,1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1,Mammal
antelope,1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1,Mammal
bass,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0,Fish
bear,1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1,Mammal
boar,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
buffalo,1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1,Mammal
calf,1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1,Mammal
carp,0,0,1,0,0,1,0,1,1,0,0,1,0,1,1,0,Fish
catfish,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0,Fish
cavy,1,0,0,1,0,0,0,1,1,1,0,0,4,0,1,0,Mammal
cheetah,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
chicken,0,1,1,0,1,0,0,0,1,1,0,0,2,1,1,0,Bird
chub,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0,Fish
clam,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,Non-insect invertebrate
crab,0,0,1,0,0,1,1,0,0,0,0,0,4,0,0,0,Non-insect invertebrate
crayfish,0,0,1,0,0,1,1,0,0,0,0,0,6,0,0,0,Non-insect invertebrate
crow,0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,0,Bird
deer,1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1,Mammal
dogfish,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,Fish
dolphin,0,0,0,1,0,1,1,1,1,1,0,1,0,1,0,1,Mammal
dove,0,1,1,0,1,0,0,0,1,1,0,0,2,1,1,0,Bird
duck,0,1,1,0,1,1,0,0,1,1,0,0,2,1,0,0,Bird
elephant,1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1,Mammal
flamingo,0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,1,Bird
flea,0,0,1,0,0,0,0,0,0,1,0,0,6,0,0,0,Insect
frog,0,0,1,0,0,1,1,1,1,1,0,0,4,0,0,0,Amphibian
frog2,0,0,1,0,0,1,1,1,1,1,1,0,4,0,0,0,Amphibian
fruitbat,1,0,0,1,1,0,0,1,1,1,0,0,2,1,0,0,Mammal
giraffe,1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1,Mammal
girl,1,0,0,1,0,0,1,1,1,1,0,0,2,0,1,1,Mammal
gnat,0,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0,Insect
goat,1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1,Mammal
gorilla,1,0,0,1,0,0,0,1,1,1,0,0,2,0,0,1,Mammal
gull,0,1,1,0,1,1,1,0,1,1,0,0,2,1,0,0,Bird
haddock,0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0,Fish
hamster,1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,0,Mammal
hare,1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,0,Mammal
hawk,0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,0,Bird
herring,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0,Fish
honeybee,1,0,1,0,1,0,0,0,0,1,1,0,6,0,1,0,Insect
housefly,1,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0,Insect
kiwi,0,1,1,0,0,0,1,0,1,1,0,0,2,1,0,0,Bird
ladybird,0,0,1,0,1,0,1,0,0,1,0,0,6,0,0,0,Insect
lark,0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0,Bird
leopard,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
lion,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
lobster,0,0,1,0,0,1,1,0,0,0,0,0,6,0,0,0,Non-insect invertebrate
lynx,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
mink,1,0,0,1,0,1,1,1,1,1,0,0,4,1,0,1,Mammal
mole,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,0,Mammal
mongoose,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
moth,1,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0,Insect
newt,0,0,1,0,0,1,1,1,1,1,0,0,4,1,0,0,Amphibian
octopus,0,0,1,0,0,1,1,0,0,0,0,0,8,0,0,1,Non-insect invertebrate
opossum,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,0,Mammal
oryx,1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1,Mammal
ostrich,0,1,1,0,0,0,0,0,1,1,0,0,2,1,0,1,Bird
parakeet,0,1,1,0,1,0,0,0,1,1,0,0,2,1,1,0,Bird
penguin,0,1,1,0,0,1,1,0,1,1,0,0,2,1,0,1,Bird
pheasant,0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0,Bird
pike,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,Fish
piranha,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0,Fish
pitviper,0,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,Reptile
platypus,1,0,1,1,0,1,1,0,1,1,0,0,4,1,0,1,Mammal
polecat,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
pony,1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1,Mammal
porpoise,0,0,0,1,0,1,1,1,1,1,0,1,0,1,0,1,Mammal
puma,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
pussycat,1,0,0,1,0,0,1,1,1,1,0,0,4,1,1,1,Mammal
raccoon,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
reindeer,1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1,Mammal
rhea,0,1,1,0,0,0,1,0,1,1,0,0,2,1,0,1,Bird
scorpion,0,0,0,0,0,0,1,0,0,1,1,0,8,1,0,0,Non-insect invertebrate
seahorse,0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0,Fish
seal,1,0,0,1,0,1,1,1,1,1,0,1,0,0,0,1,Mammal
sealion,1,0,0,1,0,1,1,1,1,1,0,1,2,1,0,1,Mammal
seasnake,0,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,Reptile
seawasp,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,Non-insect invertebrate
skimmer,0,1,1,0,1,1,1,0,1,1,0,0,2,1,0,0,Bird
skua,0,1,1,0,1,1,1,0,1,1,0,0,2,1,0,0,Bird
slowworm,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,Reptile
slug,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,Non-insect invertebrate
sole,0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0,Fish
sparrow,0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0,Bird
squirrel,1,0,0,1,0,0,0,1,1,1,0,0,2,1,0,0,Mammal
starfish,0,0,1,0,0,1,1,0,0,0,0,0,5,0,0,0,Non-insect invertebrate
stingray,0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1,Fish
swan,0,1,1,0,1,1,0,0,1,1,0,0,2,1,0,1,Bird
termite,0,0,1,0,0,0,0,0,0,1,0,0,6,0,0,0,Insect
toad,0,0,1,0,0,1,0,1,1,1,0,0,4,0,0,0,Amphibian
tortoise,0,0,1,0,0,0,0,0,1,1,0,0,4,1,0,1,Reptile
tuatara,0,0,1,0,0,0,1,1,1,1,0,0,4,1,0,0,Reptile
tuna,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,Fish
vampire,1,0,0,1,1,0,0,1,1,1,0,0,2,1,0,0,Mammal
vole,1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,0,Mammal
vulture,0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,1,Bird
wallaby,1,0,0,1,0,0,0,1,1,1,0,0,2,1,0,1,Mammal
wasp,1,0,1,0,1,0,0,0,0,1,1,0,6,0,0,0,Insect
wolf,1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1,Mammal
worm,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,Non-insect invertebrate
wren,0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0,Bird
"""
