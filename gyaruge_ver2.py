import time
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings('ignore')

a = None

print("ゲームロード中")

def T(t):
    time.sleep(t)

def H(a):
    print("ひまり : " +  str(a))
    T(2)

def I(a):
    print("自分 : "  + str(a))
    T(2)

def N(a):
    print("ナレーション : " +  str(a))
    T(2)

def IN():
    print("会話を入力してください : ")
    global a
    a = input()
    T(2)

def YN():
    print("はい、か いいえ で答えてください") # 最後修正
    IN()

def lug_model2(model,collect1, collect2):
    global eval

    global comment

    eval = []

    comment = input("会話を入力してください：")

    eval.append(max_cosine_similarity(model, comment, collect1))
    eval.append(max_cosine_similarity(model, comment, collect2))

    # print(f"paraphrase-xlm-r-multilingual-v1: commentとcollect1の類似度: {eval[0]}")
    # print(f"paraphrase-xlm-r-multilingual-v1: commentとcollect2の類似度: {eval[1]}")


def max_cosine_similarity(model, comment, collect):
    embeddings1 = model.encode(comment, convert_to_tensor=True)

    max_similarity = -1

    for item in collect:
        embeddings2 = model.encode(item, convert_to_tensor=True)

        cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
        if cosine_score > max_similarity:
            max_similarity = cosine_score

    return max_similarity

def lug_model3(model,collect1, collect2, collect3):
    global eval

    global comment

    eval = []

    comment = input("会話を入力してください：")

    eval.append(max_cosine_similarity(model, comment, collect1))
    eval.append(max_cosine_similarity(model, comment, collect2))
    eval.append(max_cosine_similarity(model, comment, collect3))

    # print(f"paraphrase-xlm-r-multilingual-v1: commentとcollect1の類似度: {eval[0]}")
    # print(f"paraphrase-xlm-r-multilingual-v1: commentとcollect2の類似度: {eval[1]}")
    # print(f"paraphrase-xlm-r-multilingual-v1: commentとcollect3の類似度: {eval[2]}")

def find_max_index(lst):
    max_value = max(lst)  
    max_index = lst.index(max_value)  
    return max_index

def lug_model_2(model,collect1, collect2):

    global eval2
    global comment
    global collectlist2

    comment = input("会話を入力してください：")

    embeddings1 = model.encode(comment, convert_to_tensor=True)
    embeddings2 = model.encode(collect1, convert_to_tensor=True)
    embeddings3 = model.encode(collect2, convert_to_tensor=True)

    cosine_score1 = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    cosine_score2 = util.pytorch_cos_sim(embeddings1, embeddings3)[0][0]

    eval2.append(cosine_score1)
    eval2.append(cosine_score2)
    collectlist2.append(collect1)
    collectlist2.append(collect1)
    collectlist2.append(collect1)

    # print(f"paraphrase-xlm-r-multilingual-v1: : {cosine_score1}"+str(comment)+"と"+str(collect1))
    # print(f"paraphrase-xlm-r-multilingual-v1: : {cosine_score2}"+str(comment)+"と"+str(collect2))

def lug_model_3(model,collect1, collect2, collect3):
    global eval3
    global comment
    global collectlist3

    collectlist3 = []
    eval3 = []

    comment = input("会話を入力してください：")

    embeddings1 = model.encode(comment, convert_to_tensor=True)
    embeddings2 = model.encode(collect1, convert_to_tensor=True)
    embeddings3 = model.encode(collect2, convert_to_tensor=True)
    embeddings4 = model.encode(collect3, convert_to_tensor=True)

    cosine_score1 = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    cosine_score2 = util.pytorch_cos_sim(embeddings1, embeddings3)[0][0]
    cosine_score3 = util.pytorch_cos_sim(embeddings1, embeddings4)[0][0]

    eval3.append(cosine_score1)
    eval3.append(cosine_score2)
    eval3.append(cosine_score3)

    collectlist3.append(collect1)
    collectlist3.append(collect2)
    collectlist3.append(collect3)

    # print(f"paraphrase-xlm-r-multilingual-v1:  {cosine_score1}"+str(comment)+"と"+str(collect1))
    # print(f"paraphrase-xlm-r-multilingual-v1:  {cosine_score2}"+str(comment)+"と"+str(collect2))
    # print(f"paraphrase-xlm-r-multilingual-v1: {cosine_score3}"+str(comment)+"と"+str(collect3))

comment = None

positive_words = ["うん", "はい", "行くよ", "賛成", "食べる", "そうだね", "いいと思う",
                  "悪くない", "やるよ", "やるやる","モテる", "モテるよ", "そうなんよ", "いる",
                  "いるよ", "それな", "だろ", "共通テスト8割だからね", ]

negative_words = ["いいえ", "いかない", "食べない", "賛成じゃない", "否定", "今回はパス",
                  "いやだ" ,"やらない","やらないかな", "モテない", "モテないよ", "そんなことないよ",
                  "全然モテない", "いまはいない", "いない", "いないよ", "しね", "カス", "うんこ",
                  "かす", "死ね", "厳しいって"]

aiamai_words = ["秘密", "ひみつ", "教えない", "さあね", "どっちでしょう", "ひまりは彼氏いるの？",
                "きみは？", "君は？", "俺の話はいいよ", "君は？", "まあまあかな", "そうかな？", ""]

model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")

collectlist3 = []
eval3 = []

collectlist2 = []
eval2 = []

eval = []



comment = None

positive_words = ["うん", "はい", "行くよ", "賛成", "食べる", "そうだね", "いいと思う", "悪くない", "やるよ", "やるやる", "モテる", "モテるよ"]

negative_words = ["いいえ", "いかない", "食べない", "賛成じゃない", "否定", "今回はパス", "いやだ" ,
                  "やらない", "やらないかな", "モテない", "モテないよ", "そんなことないよ", "まあまあかな", "全然モテない"]

model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")

collectlist3 = []
eval3 = []

collectlist2 = []
eval2 = []

eval = []


point = 0


#入学式から初一緒行動

I("今日は待ちに待った大学の入学式だ。最高の大学生を過ごすぞ。")

N("現在、入学式中、隣に座った女の子が話しかけてきた")

H("私の名前は、佐藤ひまりよろしくね！")

H("隣になったのも何かの縁だしご飯でも食べいかない？")


lug_model2(model, positive_words, negative_words)

if eval[0] > eval[1]:
    H("じゃあご飯いこっか！")
    a = "はい"
    

T(2)
if a == "はい":
    H("何食べいく？")
    restaurant1 = ["ファミレス", "ガスト", "ココス", "ご飯", "ハンバーグ",
                "パスタ", "カレーライス", "オムライス", "ピザ", "サラダ",
                "ステーキ", "フライドチキン", "エビフライ", "唐揚げ", "スープ", "ごはん", "パン", "フライドポテト", "ドリア", 
                "グラタン", "ドリンクバー", "デザート", "アイスクリーム", "ケーキ", 
                "パフェ", "ホットコーヒー", "アイスコーヒー", "紅茶", "ソフトドリンク", 
                "ビール", "ワイン", "カクテル", "キッズメニュー", "シーフード"]

    restaurant2 = ["インドカレー屋さん", "カレーライス", "カレー", "ナン"]

    restaurant3 = ["ラーメン", "醤油ラーメン", "とんこつラーメン", "醤油ラーメン",
                    "味噌ラーメン","塩ラーメン","豚骨ラーメン","魚介系ラーメン","つけ麺","混ぜそば"]


    lug_model3(model, restaurant1, restaurant2, restaurant3)

    if eval[0] > eval[1] and eval[0] > eval[2]:
        print("ひまり: じゃあ、それがありそうなファミレスに行こうか！！")

    elif eval[1] > eval[0] and eval[1] > eval[2]:

        print("ひまり: じゃあ、それがありそうなインドカレー屋さんに行こうか！！")

        point += 1
    elif eval[2] > eval[0] and eval[2] > eval[1]:

        print("ひまり: じゃあ、それがありそうなラーメンに行こうか！！")

else:
    N("ふたりは散歩を始めた")

T(1)

#ふたりの飯屋または散歩での会話

H("そういえば、君は出身はどこなの？")

IN()

if a == "茨城" or a == "茨城県" or a == "いばらき" or a == "いばらきけん":
    point = point + 2

    H("私も茨城　出身なんだ！一緒なんだね...!")

H("ところでさ、君の趣味を教えてよ！")

indoor_phrases = ["インドア派", "家好き", "室内派", "引きこもり", "家にこもるのが好きな人","読書", "映画鑑賞", "ゲーム", "料理","インドアスポーツ" ]
outdoor_phrases = ["アウトドア派", "アウトドア好き", "外出好き", "冒険好き", "野外活動が好きな人", "キャンプ", "ハイキング", "登山","テント", "焚き火", "アウトドア" , "スポーツ"]



lug_model2(model, indoor_phrases, outdoor_phrases)
if eval[0] > eval[1]:
    H("君は" + str(comment) + "が趣味なんだね！こう見えて意外とインドア派なんだね！")

    othello = 1
else:
    point += 1

    H("君は" + str(comment) + "が趣味なんだね！私もこう見えて結構アウトドア派なんだ！")

    H("私たち気が合うかもね！")
    othello = 0


H("好きな食べ物は何なの?")



H("ところで、好きな食べ物って何ー？？")

lug_model_2(model, "洋食",  "和食")

max_index = find_max_index(eval2)
print(collectlist2)
if str(collectlist2[max_index]) == "洋食":
    point = point + 1

    H(str(comment) + "が好きってことは洋食が好きなんだね！")

    H("私も洋食が好きなんだ！！")
else:
    H(str(comment) + "が好きってことは和食が好きなんだね！")

    H("私はどっちかっていうと洋食が好きなんだよねー...")


H("ところでさ、彼女とかはいたりするの...?")

N("ひまりはもじもじしながら聞いてきた。")

lug_model3(model, positive_words, negative_words, aiamai_words)


if max(eval) == eval[0]:
    H("そうなんだ、モテそうだもんね...")
    point = point -1

elif max(eval) == eval[1]:
    H("えー、もてそうなのに意外だね！")

else:
    H("えーどっちなのー？？")
    H("ごめん、プライベートすぎること聞いちゃったね...")
    point = point + 2


if othello == 0:
    H("ところでなんだけどさ、私ね、剣道サークルに興味あるんだよね！、一緒に行こうよ！！")
    N("こうして二人はサークルの体験に行くことになった")
    T(1)

else:
    H("ところでなんだけどさ、私ね、オセロサークルに興味あるんだよね！、一緒に行こうよ！！よ")
    N("こうして二人はサークルの体験に行くことになった")
    T(1)



#サークル体験
if othello == 1:

    N("二人は一緒にサークル見学に来ました 二人が来たのはボードゲームサークルです")

    H("大学のサークルってどんな感じなんだろうね？")

    H("楽しみだけどちょっと緊張するね")

    I("陽葵もこういう場は緊張するんだね そんな感じしないけど")

    H("大学の初めてのサークル見学だよ？ めちゃめちゃ緊張するよ")

    H("でも君と一緒だからちょっとは緊張しなくなってる")

    N("二人はいろいろあるボードゲームの中からオセロをやることにしました")

    H("私こういう頭使う系のゲームあんまり普段やんないんだよねー")

    H("君は普段オセロかやったりするの？")

    lug_model2(model, positive_words, negative_words)
    if eval[0] > eval[1]:
        H("すごーい 頭使うの好きなんだね ちょっと尊敬")

        H("でも負けないよー！！私地頭はいいからね（笑）")

        N("二人はオセロの対戦を楽しみました")

        H("うわーー！！負けたーー！！")

        H("さすが普段やってるだけあるね！")

        H("やっぱ思考速度の違いを感じるわ")

    else:
        H("私もー ボードゲームとかあんましやる機会ないよね")
        H("お互い初心者ってことか")

        H("私加減とかしないから 君も全力で来てね")

        H("うわーー！！負けたーー！！")

        H("結構接戦だった感あったんだけど")
        H("次やったら負けないよ")

    H("今まで話して手の思ってたけど")

    H("君って結構頭いいよね？")

    lug_model3(model, positive_words, negative_words, aiamai_words)

    if eval[0] > eval[1]:
        H("やっぱりそうだよね")

    else:
        point += 1
        H("私君に負けっちゃったもん！")

        H("頭いいよ")

        H("やっぱり頭よかったらさ...")

if othello == 0:
    N("二人は二人はアットホームな雰囲気にひかれて剣道サークルにやってきました")

    H("大学のサークルってどんな感じなんだろうね？")

    H("楽しみだけどちょっと緊張するね")

    I("陽葵もこういう場は緊張するんだね そんな感じしないけど")

    H("大学の初めてのサークル見学だよ？ めちゃめちゃ緊張するよ")

    H("でも君と一緒だからちょっとは緊張しなくなってる")

    H("私は中高で剣道やってたんだよね")

    H("君は剣道やってたの？")

    lug_model2(model, positive_words, negative_words)

    if eval[0] > eval[1]:
        H("え？まじで？一緒じゃん！！")

        H("でも負けないよー！！真剣勝負！！")

        N("二人は剣道の試合をしました 真剣勝負！！")

        H("うわーー！！負けたーー！！")

        H("一回目の面はすごかったね！")

        H("完敗だよー")

    else:
        H("へー でも道着姿けっこう様になってるよ")

        H("いろいろ教えてあげるから 楽しくやろうね")

        H("うわーー！！負けたーー！！まじかーー！！")

        H("さすがに勝てると思ったよー")

        H("次やったら負けないよ")


    H("今日練習して思ったけど...")

    H("君って運動神経めっちゃいいよね？")

    lug_model2(model, positive_words, negative_words)

    if eval[0] > eval[1]:
        H("やっぱりそうだよね")

        H("やっぱり運動できるとさ...")

    else:
        point += 1

        H("うそー 動きめっちゃ俊敏だったよ")

        H("すごかった")

        H("やっぱり運動できるとさ...")

H("モテるよね")

lug_model3(model, positive_words, negative_words, aiamai_words)
if max(eval) == eval[0]:
    point += 0

    H("へー...")

elif max(eval) == eval[1]:
    point += 1

    H("へー意外だね")

else:
    point += 3

    H("もーはぐらかさないでよお")

H("...,")

H("そういえばさ")

H("６月にこのサークルの本新歓あるらしいよ")

H("なんか地方の遊園地行くらしい")

H("なんか先輩は「ほどよくしょっぱい」？とか言ってた")

H("一緒に行こうよ")

N("二人はこのサークルに入ることにしました")

time.sleep(3)

N("本新歓当日")





#本新歓＠那須ハイランドパーク

H("とりあえず、なに乗ろうか？")


zekkyou = ["絶叫系", "ジェットコースター", "バイキング", "空中ブランコ", "逆バンジー"]
yuruyaka = ["緩やかなつ", "コーヒーカップ", "メリーゴーランド", "観覧車"]

lug_model2(model, zekkyou, yuruyaka)


if eval[0] < eval[1] :
    H("絶叫系は苦手な感じ？？じゃあ、近くにあるメリーゴーランドに乗ろう！")

    N("二人はメリーゴーランドに乗って楽しんだ")

else:
    if point > 10:
        H("やっぱり遊園地に来たら絶叫系だよね！ひとまず近くにあるジェットコースター乗ろう！！")

        N("ふたりはジェットコースターに乗った。")

        N("しかし自分は体調が悪くなってしまい、朝ごはんを戻してしまった")

        H("大丈夫？？お水買ってきたから飲んで！！ほんとに大丈夫？？")

        N("ひまりの看病のおかげで気分はだんだんと良くなった。自分は普段よりも元気になった気がした")

        point = point + 3

    else:
        pointa = point + 1

        H("やっぱり遊園地に来たら絶叫系だよね！ひとまず近くにあるジェットコースター乗ろう！！")

        N("ふたりはジェットコースターに乗って楽しんだ。")


N("楽しい時間は流れ、お昼ご飯の時間帯になった")

H("結構一通り楽しんだね！おなかぺこぺこー何食べよっか？")

restaurant1 = ["たこ焼き"]

restaurant2 = ["クレープ"]

restaurant3 = ["パスタ"]


lug_model3(model, restaurant1, restaurant2, restaurant3)

if eval[0] > eval[1] and eval[0] > eval[2]:
    print("ひまり: おっけー。じゃあ、たこ焼きいこっかー")
    point += 1
elif eval[1] > eval[0] and eval[1] > eval[2]:

    print("ひまり: じゃあ、近くにあるクレープ屋さんにいこっかー")

    
elif eval[2] > eval[0] and eval[2] > eval[1]:

    print("ひまり: じゃあ、近くにあるパスタ売ってるからそれ食べようかー")
    point += 2


T(1)

H("最後にふたりで観覧車乗ろうよ！！いこいこーーー！！")

N("ふたりは観覧車に乗ることになった")

T(1)

N("ふたりは観覧車から見える素晴らしい景色を見ながらゆっくりと進む時間を楽しんだ")

N("しかし、、、、")

N("ここで事件は起こった")

N("観覧車が途中で止まってしまった")

N("アナウンスが流れ、その内容は、、観覧車が故障したため、現在停止しているということだった")

N("ひまりは恐怖のあまり震えだしてしまうほど、おびえていた")

N("ここで、一緒にいるのはあなただけです。ひまりに慰めの言葉をかけてあげるのです！！")




maxcollect = ["大丈夫、僕がついてるよ"]
midiumcollect = ["観覧車なんてそうそう落ちるものじゃないから大丈夫だよ"]
mincollect = ["なんでそんなおびえてるの", "しね", "やば", "なにもできない"]

lug_model3(model, maxcollect, midiumcollect, mincollect)

if eval[0] > eval[1] and eval[0] > eval[2]:
    N("声をかけるとひなみの震えは止まり、安心したような顔をしていた")

    point = point + 4

elif eval[1] > eval[0] and eval[1] > eval[2]:
    N("ひなみは少し安心したような様子だ")

    point += 2

    
elif eval[2] > eval[0] and eval[2] > eval[1]:
    N("ひなみの様子は変わらなかった") 

N("アナウンスとともに観覧車は動き出し、ふたりは安どした")

H("ひとときはどうなるかと思ったけど、私たち生きてる！！君がついていてくれたおかげだね！")

N("色んなことがあったが楽しい一日となった。")

T(3)

N("時は過ぎ、ふたりは地元のお祭りに参加することになった")

N("ぼくは、この日にすべてをかけることを決めていた")

N("ひなみの浴衣姿はとてもきれいで、未来のことが考えられなくなるほどだった")

N("ふたりは射的をしたり、お好み焼きを食べたりしてこれには代えられない時間を過ごした")

N("花火があがり、その花火は子供のころ見た景色とは違っていて、")

N("大人とも子供とも言えない現在の自分には足の裏がかゆくなるような感覚を残しながら")

N("あっという間に花火が終わった。")

rate = 100*point/22
N("いろいろなことがあったが楽しい１日となった")

H("今日は楽しかったね")

IN()

H("...")

H("私たちここ半年で結構仲良くなったよね..")

N("二人はいい雰囲気")

N("ここで告白しましょう")

IN()
if point >= 18:
    H("え、うそ....")

    H("うれしい")

    H("でも、君は完璧すぎて私には釣り合わないよ....")

    N("ひまりはそうつぶやくと泣き出してしまった")

    IN()

    H("ほんと？？ほんとにこんな私でいいの？？")

    IN()

    H("あリがとう")

    H("私も君のことが好きです...")

    H("これからよろしくお願いします")

    print("HAPPY END")

elif point <18 and point >= 11:
    H("え、ほんと？")

    H("ありがとう")

    H("私も君のことが好きです...")

    H("これからよろしくお願いします")

    print("HAPPY END")

else:
    H("あ、ありがとう")

    H("気持ちはうれしいんだけど...")

    H("君とは友達でいたいな")

    H("ごめんね")

    print("GAME　fucking OVER")

N("あなたの得点率は、")
N(str(rate))
N("でした。")
N("いったんお疲れさまでした")
N("得点は、")
N(str(point))