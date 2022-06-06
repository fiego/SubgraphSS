
import fasttext
from model.fastText import fastText

if __name__ == '__main__':
    # fastText = fastText()  ## 加载模型
    # mode = fastText.train()

    model = fasttext.load_model("./checkpoint/fastText/fastText_300d_200e.model")

    print(model.get_word_vector("the").shape)
    print(model.get_word_vector("the").reshape(1, -1).shape)



