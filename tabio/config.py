in_dir = '/home/amit/experiments/tabio/finalset/all_links'


test_hashes = [
#Total:
'744df57024cad2d7d5ce08b7bb44322e',
'5684a5b404ca5e91dfecb791e8a901a3',

#Technip:
'623b8e806a4b324c0ed65ea2812a70fa',
'4bfdba590339392cd2504d8bf5b3ad93',

#Schlumberger:
'6cc024556cf81d685fd045684a951210',

#Sapiem:
'12682288207b119847c5d550bf92df37',
'd11fc8e706d28485e0045822c3601afb',

#IFP:
'30039718573b20e0bbc5f87121890534',
'37ebadf81ad8bc694c8a3988582b8d2d',
'2ec6960ba691c6dc7880cabeafce2129',
'32c1eb91db6a924e1a30a7c78bdd4ab0',
'128a980c236d85c867f81557e34cb7f3',
'20bceabf938737361888dc718db02c8c',
'42c8f8a90bc9a6e37f9de4b055421ba8',
'1d3a3d5e43d75aea35aafcaf09687480',
'072e99c74d60aa6c9fb3fd5992b8c239',
'8b19c56c096dbb5453e8993329914442',
]


# classes = ['Author', 'Equation', 'FigureCaption', 'FigureText', 'FrameSpareMulticolumn', 'Heading', 'PageFooter', 'PageHeader', 'PageNumber', \
#    'Paragraph', 'References', 'Sparse', 'Subtitle', 'TableCaption', 'TableFooter', 'TableSparseColumnHeader', 'TableSparseMulticolumn', 'TableSuperHeader', 'Title']
class_map = {'Author':"Else", 'Equation':"Else", 'FigureCaption':"Else", 'FigureText':"Else", 'FrameSpareMulticolumn':"Table", 'Heading':"Else", 'PageFooter':"Else", 'PageHeader':"Else", 'PageNumber':"Else", 'Paragraph':"Else", 'References':"Else", 'Sparse':"Else", 'Subtitle':"Else", 'TableCaption':"Else", 'TableFooter':"Else", 'TableSparseColumnHeader':"Table", 'TableSparseMulticolumn':"Table", 'TableSuperHeader':"Table", 'Title':"Else",'TableSpareMulticolumn':"Table","FrameSparseMulticolumn":"Table","TableSparseOther":"Table","TableCaptionContd":"Else","TableSparse":"Table","TableSpareColumnHeader":"Table"}

# If text in the training set is not labeled, then optionally
# treat this text as the class specified below.
#
# To ignore unlabeled text during training, set to `None`
unlabeled_class = None

mapped_classes = ["Table","Else"]
classes = mapped_classes

# Normally, we need to detect double column portions of a page
# so the line classifier can be ran on both the left and right
# side if there are two columns.
#
# If it is known that none of the pages have multiple columns
# then column detection can be turned off, and all lines will
# be treated as if they are full width
#
# Disabling column detection speeds up processing and prevents
# misclassifying a single column as double if it is known that
# there are no double columns
enable_column_detection = True

col_classes = {"SingleColumn":0,"DoubleColumn":1,"None":2,"DoublColumn":1,None:2}
col_class_inference = {0:"SingleColumn",1:"DoubleColumn",2:"SingleColumn"}
tune = [0.4,-0.4]

# manually labeled data can be marked with column information as well as a class
# this function returns (column_class, class)
def interpret_label(label):
    if not enable_column_detection:
        return ('SingleColumn', label)
    column_class, line_class = label.split("-")
    return (column_class, line_class)
