from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from django.template.loader import render_to_string
from django.views import View
from django.conf import settings
import numpy as np

class Index(View):
    template_name = 'searcher/index.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        search_terms = request.POST.get('search_terms')
        top_n = int(request.POST.get('top_n'))
        search_terms_array = search_terms.split(" ")
        sum_of_terms = []
        for term in search_terms_array:
            word_vector = settings.TERM_SEARCH_ENGINE.get_word_vector(term)
            sum_of_terms.append(word_vector)

        average_word = np.average(sum_of_terms, axis=0)

        similar_terms = settings.TERM_SEARCH_ENGINE._similarity_query(average_word, top_n)
        used_terms = []
        for (word, sim) in similar_terms:
            if({'word':word,'sim':sim} not in used_terms):
                used_terms.append({'word':word, 'sim':sim})

        term_results_template = render_to_string('searcher/term_results.html', {'used_terms': used_terms})

        found_images = []
        for (word, sim) in similar_terms:
            if word in settings.IMAGE_RETRIEVED_TAGS:
                for image in settings.IMAGE_RETRIEVED_TAGS[word]:
                    if({'image':image} not in found_images):
                        found_images.append({'image':image})

        print(found_images)
        images_results_template = render_to_string('searcher/results.html',{'found_images':found_images})

        context = {'term_results' : term_results_template, 'image_results' : images_results_template}
        return JsonResponse(context)




