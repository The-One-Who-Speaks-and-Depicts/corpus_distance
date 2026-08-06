"""Contains mocks for the data_preprocessing module functions.

TEXT_FOR_TOPIC_MODELLING_POLACK (str): the text to undergo topic modelling for key "Polack".
TOPIC_MODELLED_TOKENS_POLACK (list[str]): the result of topic modelling for
TEXT_FOR_TOPIC_MODELLING_POLACK.
TEXT_FOR_TOPIC_MODELLING_NOVGOROD (str): the text to undergo topic modelling for key "Novgorod".
TOPIC_MODELLED_TOKENS_NOVGOROD (list[str]): the result of topic modelling for
TEXT_FOR_TOPIC_MODELLING_NOVGOROD.
INCORRECT_ARGS_BUILD_TOPIC_WORDS_FOR_LECT (list): the arguments that are not allowed in the
build_topic_words_for_lect function.
INCORRECT_ARGS_SAVE_TOPIC_MODELLING_RESULTS (list): the arguments that are not allowed in the
save_topic_modelling_results function.
TOPIC_MODELLING_CORRECT_RESULT (dict): the correct inputs and outputs of the module functions.
"""

from pandas import DataFrame

from corpus_distance.data_preprocessing import topic_modelling as tm

TEXT_FOR_TOPIC_MODELLING_POLACK = "с азъ , андрѣи данильѥвичь . ажо ми сѧ оучинить не быть оу " \
"полотськѣ , даю сельце на просмоужьци , свою очину и дѣдиноу , стои троци оцю своѥму i матери " \
"и своѥму племени нa памѧть и собе на памѧть . ажо oуступитьсѧ которыи" \
" кнѧзь или вельможа , да судить ѥму стаѧ троца на страшьномъ судѣ , да будеть проклѧтъ ."
TOPIC_MODELLED_TOKENS_POLACK = [',', '.', 'i', 'oуступитьсѧ', 'ажо',
                         'азъ', 'андрѣи', 'будеть', 'быть', 'вельможа']

TEXT_FOR_TOPIC_MODELLING_NOVGOROD = "Се язъ князь Ярославъ Володимѣричь , сгадавъ с посадникомъ" \
" с Мирошкою и с тысяцкымь Яковомъ и съ всѣми Новгородьци , потвердихомъ мира старого ," \
" с посломь Арбѹдомь и съ всѣми Нѣмецкыми сыны , и съ Гты , и съ всемь Латиньскымь языкомь , " \
"послалъ ѥсмь посла своего Григѹ на сеи правдѣ . Первое : ходити Новгородцю послу и всѧкомѹ " \
"Новгорожцю в миръ в Нѣмечьскѹ землю и на Гъцкъ берегъ . тако же ходити Нѣмьчьмь и Гтѧномъ" \
" в Новъгородъ безъ пакости не ѡбидимъ инеым же . Аче бѹдетъ сѹдъ кнѧзю Новгороцкъмѹ Новѣгородѣ" \
" или Немецкъмѹ въ Немчьхъ . а в томь мирѹ ити гостю домовь бес пакости ;" \
" а кого Богъ поставить . кнѧзѧ а съ тѣмъ мира потвердить . любо ли землѧ без мирѹ станеть ." \
" А ѡже ѹбьють Новгородца посла за моремъ или Нѣмецкыи посолъ Новѣгородѣ ." \
" то за тѹ головѹ ~к~ гривенъ серебра . А ѡже ѹбьють купчинѹ Новгоролца" \
" или Нѣмчина кѹпчинѹ Новѣгородѣ . то за тѹ головѹ ~i~ гривенъ серебра ." \
" А ѡже мѹжа свѧжють без віны • то ~ві~ гривенъ за соромъ старыхъ кѹнъ . Ѡже ѹдарѧть мѹжа" \
" ѡрѹжеѥмь любо коломъ , то ~г~ гривенъ за ранѹ старыѥ . Ѡже ѹпьхньть любо матель роздрьть ," \
" то ~г~ гривны старыѥ . Ѡже пошибаѥть мужескѹ женѹ любо дчьрь . то кнѧзю ~м~" \
" гривенъ ветхъми кѹнами . Ѡже съгренеть чюжеѥ женѣ повои с головы или дщьри . явитсѧ" \
" простоволоса ~г~гривенъ старыѥ за соромъ . Ѡже тѧжа родитсѧ бес крови . снидѵтсѧ послѵси" \
" и Рѵсь и Нѣмци . то вергѹть жеребеѥ . комѵ сѧ выимьть . ротѣ шьдъ свою правдѹ възмѹть ." \
" Ѡже ѥмати скотъ Варѧгѹ на Рѹсинѣ или Рѵсинѹ на Варѧзѣ . а сѧ ѥго заприть ." \
" то ~ві~ мѵжь послѹхы идеть ротѣ възметъ своѥ . Ѡже родитсѧ тѧжа в Нѣмцехъ Новгородцю ." \
" то рѵбежа не творити . на дрѵгоѥ лѣто жаловати . Ѡже не правѧть . то кнѧзю явѧ и людемъ" \
" взѧти своѥ ѹ гости . ѡже тѧжа родитсѧ в Новѣгородѣ . Ѡже тѧжа родить въ іноѥ зѥмли" \
" в Рѵскыхъ городѣхъ . то ѹ тѣхъ своѥ тѧже прашати . искати Новѵгородѵ не надобе ." \
" А тѧжа на городы а Нѣмчинъ свободь и Новгороци . Ѡже придетъ съ своѥи лодьи" \
" в Нѣмецкои домовь . аче самъ не поидьть в неи ѡпѧть мѵжь . дастъ кърмьникѹ . Нѣмчина не" \
" сажати в погребъ Новѣородѣ ни Новгородца въ Нѣмцьхъ . нъ ѥмати своѥ ѹ виновата ." \
" Ѡже кто робѵ повержьть насильѥмь а не соромить . то за ѡбидѵ гривна ." \
" пакы ли соромить собѣ свободна •" \
" Ѡже ѹбьють таль или попъ Новгороцкоѥ или Нѣмецкъѥ Новѣгородѣ ." \
" то ~к~ гривенъ серебра за головѹ . "
TOPIC_MODELLED_TOKENS_NOVGOROD = ['', ',', '.', ':', ';', '~i~', '~ві~', '~г~',
                                  '~г~гривенъ', '~к~', 'в', 'за', 'и',
                                  'или', 'на', 'не', 'то', 'Ѡже']


INCORRECT_ARGS_BUILD_TOPIC_WORDS_FOR_LECT = [
    '',
    ' ',
    'string',
    -1,
    0.5,
    3,
    True,
    [],
    ['lect'],
    {},
    {
        'lect': 'Croatian'
    },
    DataFrame(
        {
            '1': ['Croatian', 'token dummy token'],
            '2': ['Slovenian', 'token']
        }.items(), columns = ['lang', 'text']
        ),
    DataFrame(
        {
            '1': ['Croatian', 'token dummy token'],
            '2': ['Slovenian', 'token']
        }.items(), columns = ['lect', 'string']
        ),
    tm.LDAParams(num_topics=0),
    tm.LDAParams(epochs=0),
    tm.LDAParams(passes=0)
]


INCORRECT_ARGS_SAVE_TOPIC_MODELLING_RESULTS = [
    '',
    ' ',
    'string',
    0,
    1,
    -1,
    0.5,
    True,
    [],
    ['lect'],
    {},
    {
        'lect': 'Croatian'
    },
    DataFrame.from_dict(
        {
            '1': ['Croatian', 'token dummy token', 'token dummy token'],
            '2': ['Slovenian', 'token', 'token']
        }, orient='index', columns = ['lang', 'text', 'text_topic_normalised']
        ),
    DataFrame.from_dict(
        {
            '1': ['Croatian', 'token dummy token', 'token dummy token'],
            '2': ['Slovenian', 'token', 'token']
        }, orient='index', columns = ['lect', 'string', 'text_topic_normalised']
        ),
    DataFrame.from_dict(
        {
            '1': ['Croatian', 'token dummy token', 'token dummy token'],
            '2': ['Slovenian', 'token', 'token']
        }, orient='index', columns = ['lect', 'string', 'text']
        )
]

TOPIC_MODELLING_CORRECT_ARGS = {
    "initial": DataFrame.from_dict(
        {
            '1': ["Polack", TEXT_FOR_TOPIC_MODELLING_POLACK],
            '2': ["Novgorod", TEXT_FOR_TOPIC_MODELLING_NOVGOROD]
        }, orient='index', columns = ['lect', 'text']
    ),
    "topics": {
        "Polack": TOPIC_MODELLED_TOKENS_POLACK,
        "Novgorod": TOPIC_MODELLED_TOKENS_NOVGOROD
    },
    "dir": "exp_1",
    "result_not_substituted": DataFrame.from_dict(
        {
            '1': [
                "Polack",
                TEXT_FOR_TOPIC_MODELLING_POLACK,
                TEXT_FOR_TOPIC_MODELLING_POLACK
                ],
            '2': [
                "Novgorod",
                TEXT_FOR_TOPIC_MODELLING_NOVGOROD,
                TEXT_FOR_TOPIC_MODELLING_NOVGOROD
                ]
        }, orient='index', columns = ['lect', 'text', 'text_topic_normalised']
    )
}

SHINGLE_FREQUENCIES = {'Novgorod': [43,
              437,
              1027,
              1196,
              1127,
              952,
              779,
              619,
              494,
              431,
              384,
              351,
              336,
              330,
              329],
 'Polack': [52,
            434,
            1006,
            1171,
            1071,
            888,
            728,
            596,
            503,
            437,
            391,
            367,
            353,
            350,
            350],
 'Smolensk': [48,
              429,
              1065,
              1296,
              1232,
              1052,
              878,
              709,
              596,
              508,
              453,
              414,
              401,
              396,
              396]
}
