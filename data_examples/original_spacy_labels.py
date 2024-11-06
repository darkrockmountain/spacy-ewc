# generated using utils.generate_spacy_entities
original_spacy_labels = [
    (
        "John bought a new car yesterday.",
        {"entities": [(0, 4, "PERSON"), (22, 31, "DATE")]},
    ),
    (
        "Microsoft announced its new software in San Francisco.",
        {"entities": [(0, 9, "ORG"), (40, 53, "GPE")]},
    ),
    (
        "The meeting is scheduled for 3 p.m. on Friday.",
        {"entities": [(29, 35, "TIME"), (39, 45, "DATE")]},
    ),
    (
        "Sarah earned $500 in tips last week.",
        {"entities": [(0, 5, "PERSON"), (14, 17, "MONEY"), (26, 35, "DATE")]},
    ),
    (
        "Amazon plans to open a new warehouse in Seattle.",
        {"entities": [(0, 6, "ORG"), (40, 47, "GPE")]},
    ),
    (
        "On June 15th, the festival will take place in Central Park.",
        {"entities": [(3, 12, "DATE"), (46, 58, "LOC")]},
    ),
    (
        "Alice went to the dentist at 10:30 a.m. today.",
        {"entities": [(0, 5, "PERSON"), (29, 39, "TIME"), (40, 45, "DATE")]},
    ),
    (
        "The company raised $2 million in funding.",
        {"entities": [(19, 29, "MONEY")]},
    ),
    (
        "Tom visited Paris in July.",
        {"entities": [(0, 3, "PERSON"), (12, 17, "GPE"), (21, 25, "DATE")]},
    ),
    (
        "The restaurant opens at 8:00 a.m. every day.",
        {"entities": [(24, 33, "TIME"), (34, 43, "DATE")]},
    ),
    (
        "Lucy paid $45 for her meal at the Italian restaurant.",
        {"entities": [(0, 4, "PERSON"), (11, 13, "MONEY"), (34, 41, "NORP")]},
    ),
    (
        "Facebook hired 100 new employees last month.",
        {"entities": [(15, 18, "CARDINAL"), (33, 43, "DATE")]},
    ),
    (
        "On Christmas Eve, the store closed early.",
        {"entities": [(3, 16, "DATE")]},
    ),
    (
        "The conference will be held in New York on April 12.",
        {"entities": [(31, 39, "GPE"), (43, 51, "DATE")]},
    ),
    (
        "Jake owes $150 to the electric company.",
        {"entities": [(0, 4, "ORG"), (11, 14, "MONEY")]},
    ),
    (
        "She travels to Japan every year in October.",
        {"entities": [(15, 20, "GPE"), (21, 31, "DATE"), (35, 42, "DATE")]},
    ),
    (
        "The rent is due by the 1st of each month.",
        {"entities": [(19, 40, "DATE")]},
    ),
    (
        "Harvard University has a beautiful campus.",
        {"entities": [(0, 18, "ORG")]},
    ),
    (
        "The meeting was postponed to 2 p.m. on Wednesday.",
        {"entities": [(29, 35, "TIME"), (39, 48, "DATE")]},
    ),
    (
        "Carlos bought a ticket to the concert for $75.",
        {"entities": [(0, 6, "PERSON"), (43, 45, "MONEY")]},
    ),
    (
        "They plan to launch the product in December.",
        {"entities": [(35, 43, "DATE")]},
    ),
    (
        "Apple's stock reached $150 per share.",
        {"entities": [(0, 5, "ORG"), (23, 26, "MONEY")]},
    ),
    (
        "Emma's birthday is on the 22nd of March.",
        {"entities": [(0, 4, "ORG"), (22, 39, "DATE")]},
    ),
    (
        "The event starts at noon and ends at 6 p.m.",
        {"entities": [(20, 24, "TIME"), (37, 43, "TIME")]},
    ),
    (
        "Global Corp expanded its business to Europe.",
        {"entities": [(0, 11, "ORG"), (37, 43, "LOC")]},
    ),
    (
        "Rachel spent $200 on groceries last weekend.",
        {"entities": [(0, 6, "PERSON"), (14, 17, "MONEY"), (31, 43, "DATE")]},
    ),
    (
        "The museum is located in downtown Chicago.",
        {"entities": [(34, 41, "GPE")]},
    ),
    (
        "The movie premiere will be on August 1st.",
        {"entities": [(30, 40, "DATE")]},
    ),
    (
        "James donated $500 to the animal shelter.",
        {"entities": [(0, 5, "PERSON"), (15, 18, "MONEY")]},
    ),
    (
        "The shop opens at 9:00 a.m. on Saturdays.",
        {"entities": [(18, 27, "TIME"), (31, 40, "DATE")]},
    ),
    (
        "The CEO will speak at the conference in May.",
        {"entities": [(40, 43, "DATE")]},
    ),
    (
        "Liam took a flight to Los Angeles last night.",
        {"entities": [(0, 4, "PERSON"), (22, 33, "GPE"), (34, 44, "TIME")]},
    ),
    (
        "The store had a sale on Monday for 50% off.",
        {"entities": [(24, 30, "DATE"), (35, 38, "PERCENT")]},
    ),
    (
        "Sophie works at Google as a software engineer.",
        {"entities": [(0, 6, "ORG"), (16, 22, "ORG")]},
    ),
    (
        "The class starts at 8:15 a.m. on weekdays.",
        {"entities": [(20, 29, "TIME"), (33, 41, "DATE")]},
    ),
    (
        "The charity raised $10,000 in donations.",
        {"entities": [(20, 26, "MONEY")]},
    ),
    (
        "Mia went to the gym at 6:00 a.m. today.",
        {"entities": [(0, 3, "PERSON"), (23, 32, "TIME"), (33, 38, "DATE")]},
    ),
    (
        "The bank charged a fee of $25 for the transfer.",
        {"entities": [(27, 29, "MONEY")]},
    ),
    ("The wedding is set for October 10th.", {"entities": [(23, 35, "DATE")]}),
    (
        "They bought a house in Miami for $300,000.",
        {"entities": [(23, 28, "GPE"), (34, 41, "MONEY")]},
    ),
    (
        "The school opens at 7:30 a.m. every morning.",
        {"entities": [(20, 29, "TIME")]},
    ),
    (
        "Jack received a bonus of $1,000.",
        {"entities": [(0, 4, "PERSON"), (26, 31, "MONEY")]},
    ),
    ("The train leaves at 9:45 a.m. sharp.", {"entities": [(20, 29, "TIME")]}),
    (
        "Lucas and Emma traveled to Germany in spring.",
        {
            "entities": [
                (0, 5, "ORG"),
                (10, 14, "PERSON"),
                (27, 34, "GPE"),
                (38, 44, "DATE"),
            ]
        },
    ),
    ("The ticket costs $12 for adults.", {"entities": [(18, 20, "MONEY")]}),
    (
        "They will announce the winner on June 3rd.",
        {"entities": [(33, 41, "DATE")]},
    ),
    (
        "Ella has a dentist appointment at 4:15 p.m.",
        {"entities": [(0, 4, "ORG"), (34, 43, "TIME")]},
    ),
    (
        "The park closes at 9:00 p.m. during summer.",
        {"entities": [(19, 28, "TIME"), (36, 42, "DATE")]},
    ),
    (
        "The concert will take place in Madison Square Garden.",
        {"entities": [(31, 52, "FAC")]},
    ),
    (
        "They booked a vacation to Bali in September.",
        {"entities": [(26, 30, "GPE"), (34, 43, "DATE")]},
    ),
    ("The cafe charges $3 for a coffee.", {"entities": [(18, 19, "MONEY")]}),
    ("His flight departs at 11:00 a.m.", {"entities": [(22, 32, "TIME")]}),
    (
        "The meeting starts at 10:00 on Thursday.",
        {"entities": [(22, 39, "TIME")]},
    ),
    (
        "Zara hired 50 new employees last quarter.",
        {"entities": [(0, 4, "ORG"), (11, 13, "CARDINAL"), (28, 40, "DATE")]},
    ),
    (
        "She pays $50 for her gym membership monthly.",
        {"entities": [(10, 12, "MONEY"), (36, 43, "DATE")]},
    ),
    ("The bank is located on Fifth Avenue.", {"entities": [(23, 35, "FAC")]}),
    (
        "The marathon is scheduled for October 3rd.",
        {"entities": [(30, 41, "DATE")]},
    ),
    (
        "The store is having a sale on April 5th.",
        {"entities": [(30, 39, "DATE")]},
    ),
    (
        "Ryan's flight landed at 6:00 a.m. today.",
        {"entities": [(0, 4, "PERSON"), (24, 33, "TIME"), (34, 39, "DATE")]},
    ),
    (
        "The zoo opens at 10:00 a.m. on weekends.",
        {"entities": [(17, 27, "TIME")]},
    ),
    ("His car was towed for $200.", {"entities": [(23, 26, "MONEY")]}),
    (
        "The theater has a show at 7:30 p.m. daily.",
        {"entities": [(26, 41, "TIME")]},
    ),
    (
        "Anna works at the United Nations.",
        {"entities": [(0, 4, "PERSON"), (14, 32, "FAC")]},
    ),
    ("They sold their house for $450,000.", {"entities": [(27, 34, "MONEY")]}),
    (
        "The school charges $300 per semester for books.",
        {"entities": [(20, 23, "MONEY")]},
    ),
    (
        "The game starts at 4:30 p.m. tomorrow.",
        {"entities": [(19, 28, "TIME"), (29, 37, "DATE")]},
    ),
    ("The museum entry fee is $15.", {"entities": [(25, 27, "MONEY")]}),
    (
        "They launched the product on March 18.",
        {"entities": [(29, 37, "DATE")]},
    ),
    (
        "The gala will be held in the Grand Ballroom.",
        {"entities": [(25, 43, "FAC")]},
    ),
    (
        "He bought a ticket to the game for $60.",
        {"entities": [(36, 38, "MONEY")]},
    ),
    (
        "The restaurant in Miami opens at 6:00 p.m.",
        {"entities": [(18, 23, "GPE"), (33, 42, "TIME")]},
    ),
    ("Her salary was increased by $5,000.", {"entities": [(29, 34, "MONEY")]}),
    (
        "They renewed their lease for another year.",
        {"entities": [(29, 41, "DATE")]},
    ),
    (
        "The pool opens at 8 a.m. during summer.",
        {"entities": [(18, 24, "TIME"), (32, 38, "DATE")]},
    ),
    (
        "The workshop starts on September 14th.",
        {"entities": [(23, 37, "DATE")]},
    ),
    ("His flight arrives at noon.", {"entities": [(22, 26, "TIME")]}),
    ("Their mortgage is $1,200 per month.", {"entities": [(19, 24, "MONEY")]}),
    (
        "The show will be in the Broadway Theater.",
        {"entities": [(20, 40, "ORG")]},
    ),
    ("Her office is on the 20th floor.", {"entities": [(21, 25, "ORDINAL")]}),
    (
        "They signed a contract on April 20th.",
        {"entities": [(26, 36, "DATE")]},
    ),
    (
        "The rent is due by the 5th of each month.",
        {"entities": [(19, 40, "DATE")]},
    ),
    (
        "The gala will be hosted by ABC Network.",
        {"entities": [(27, 38, "ORG")]},
    ),
    ("She spent $75 on a pair of shoes.", {"entities": [(11, 13, "MONEY")]}),
    (
        "The company raised $5 million in investment.",
        {"entities": [(19, 29, "MONEY")]},
    ),
    (
        "The school semester begins in August.",
        {"entities": [(30, 36, "DATE")]},
    ),
    (
        "She travels to Mexico every December.",
        {"entities": [(15, 21, "GPE"), (28, 36, "DATE")]},
    ),
    (
        "He works at a hospital in New York City.",
        {"entities": [(26, 39, "GPE")]},
    ),
    ("They went to dinner at 8:30 p.m.", {"entities": [(23, 32, "TIME")]}),
    (
        "The bill was $90 for the whole group.",
        {"entities": [(14, 16, "MONEY")]},
    ),
    ("The convention will be held in July.", {"entities": [(31, 35, "DATE")]}),
    (
        "Their anniversary is on February 14th.",
        {"entities": [(24, 37, "DATE")]},
    ),
    ("The ticket cost $20 per person.", {"entities": [(17, 19, "MONEY")]}),
    ("They attended the ceremony at 5 p.m.", {"entities": [(30, 36, "TIME")]}),
    ("Her favorite restaurant is in Boston.", {"entities": [(30, 36, "GPE")]}),
    ("He paid $15 for the parking fee.", {"entities": [(9, 11, "MONEY")]}),
    (
        "The library opens at 9:00 a.m. daily.",
        {"entities": [(21, 30, "TIME"), (31, 36, "DATE")]},
    ),
    (
        "Their flight to Rome leaves at 10:15 a.m.",
        {"entities": [(16, 20, "GPE"), (31, 41, "TIME")]},
    ),
    ("The course fee is $500.", {"entities": [(19, 22, "MONEY")]}),
    (
        "They moved to San Diego last year.",
        {"entities": [(14, 23, "GPE"), (24, 33, "DATE")]},
    ),
    ("The lecture will be at the main hall.", {"entities": []}),
    ("Her birthday is on December 25th.", {"entities": [(19, 32, "DATE")]}),
    ("The plane landed at 7:30 p.m.", {"entities": [(20, 29, "TIME")]}),
    ("They bought a laptop for $1,200.", {"entities": [(26, 31, "MONEY")]}),
    ("The game begins at 7 p.m.", {"entities": [(19, 25, "TIME")]}),
    (
        "They plan to visit the museum on June 20.",
        {"entities": [(33, 40, "DATE")]},
    ),
    ("The car cost him $25,000.", {"entities": [(18, 24, "MONEY")]}),
    ("Her office is on the 5th Avenue.", {"entities": [(21, 31, "FAC")]}),
    (
        "The graduation is scheduled for May 5th.",
        {"entities": [(32, 39, "DATE")]},
    ),
    (
        "He works for the United States Postal Service.",
        {"entities": [(13, 45, "ORG")]},
    ),
    ("Their wedding was in Las Vegas.", {"entities": [(21, 30, "GPE")]}),
    ("The class begins at 11:00 a.m.", {"entities": [(20, 30, "TIME")]}),
    ("They received a grant of $10,000.", {"entities": [(26, 32, "MONEY")]}),
    ("Her appointment is at noon.", {"entities": [(22, 26, "TIME")]}),
    ("They donated $100 to the food bank.", {"entities": [(14, 17, "MONEY")]}),
    ("The store is on the second floor.", {"entities": [(20, 26, "ORDINAL")]}),
    (
        "The parade is on the first Saturday in June.",
        {
            "entities": [
                (21, 26, "ORDINAL"),
                (27, 35, "DATE"),
                (39, 43, "DATE"),
            ]
        },
    ),
    (
        "They rent an apartment for $900 monthly.",
        {"entities": [(28, 31, "MONEY"), (32, 39, "DATE")]},
    ),
    (
        "Her exam is on Wednesday at 2 p.m.",
        {"entities": [(15, 24, "DATE"), (28, 34, "TIME")]},
    ),
    ("The shop charges $10 for shipping.", {"entities": [(18, 20, "MONEY")]}),
    (
        "They will visit the park on Saturday.",
        {"entities": [(28, 36, "DATE")]},
    ),
    ("The concert costs $50 per ticket.", {"entities": [(19, 21, "MONEY")]}),
    (
        "Her job is located in downtown Denver.",
        {"entities": [(31, 37, "GPE")]},
    ),
    (
        "The lab opens at 6 a.m. every weekday.",
        {"entities": [(17, 23, "TIME")]},
    ),
    # Camel Case company names for spacy Camel Case recognition.
    (
        "eBay hosts auctions for rare antiques from England, often attracting over 100,000 visitors daily to bid online.",
        {
            "entities": [
                (0, 4, "ORG"),
                (43, 50, "GPE"),
                (74, 81, "CARDINAL"),
                (91, 96, "DATE"),
            ]
        },
    ),
    (
        "Released by Apple in 2023, the iPhone 15 is available in major cities like New York, London, and Tokyo, with features like 5G support.",
        {
            "entities": [
                (12, 17, "ORG"),
                (21, 25, "DATE"),
                (75, 83, "GPE"),
                (85, 91, "GPE"),
                (97, 102, "GPE"),
                (123, 124, "CARDINAL"),
            ]
        },
    ),
    (
        "On YouTube, users can watch cooking tutorials by Chef Gordon Ramsay, published every Wednesday at 5 PM.",
        {
            "entities": [
                (3, 10, "ORG"),
                (54, 67, "PERSON"),
                (85, 94, "DATE"),
                (98, 102, "TIME"),
            ]
        },
    ),
    (
        "FedEx delivers packages from Los Angeles to Paris, with guaranteed overnight shipping for business clients.",
        {
            "entities": [
                (0, 5, "GPE"),
                (29, 40, "GPE"),
                (44, 49, "GPE"),
                (67, 76, "TIME"),
            ]
        },
    ),
    (
        "MasterCard partnered with Amazon in 2022 to offer exclusive discounts on Prime Day for customers in the United States.",
        {
            "entities": [
                (0, 10, "ORG"),
                (26, 32, "ORG"),
                (36, 40, "DATE"),
                (73, 82, "DATE"),
                (100, 117, "GPE"),
            ]
        },
    ),
    (
        "LinkedIn is widely used by professionals like Sarah Johnson, a software engineer in San Francisco, to connect with companies like Google and Microsoft.",
        {
            "entities": [
                (0, 8, "ORG"),
                (46, 59, "PERSON"),
                (84, 97, "GPE"),
                (130, 136, "ORG"),
                (141, 150, "ORG"),
            ]
        },
    ),
    (
        "HarperCollins released a new novel by Margaret Atwood in September, which was quickly sold out in bookstores across Canada.",
        {
            "entities": [
                (38, 53, "PERSON"),
                (57, 66, "DATE"),
                (116, 122, "GPE"),
            ]
        },
    ),
    (
        "BioWare's latest game, Dragon Age: Origins, was developed in collaboration with artists in Tokyo and launched at Comic-Con in San Diego.",
        {
            "entities": [
                (0, 7, "ORG"),
                (23, 42, "WORK_OF_ART"),
                (91, 96, "GPE"),
                (113, 122, "PERSON"),
                (126, 135, "GPE"),
            ]
        },
    ),
]
