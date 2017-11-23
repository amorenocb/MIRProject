from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from django.template.loader import render_to_string
from django.views import View
from django.conf import settings

class Index(View):
    template_name = 'searcher/index.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        search_terms = request.POST.get('search_terms')
        top_n = int(request.POST.get('top_n'))
        search_terms_array = search_terms.split(" ")
        similar_terms = []
        for term in search_terms_array:
            results = settings.TERM_SEARCH_ENGINE.most_similar(term.lower(), top_n+1)
            similar_terms.extend(results)

        used_terms = []
        for word, prob in similar_terms:
            used_terms.append({'word':word, 'prob':prob})

        term_results_template = render_to_string('searcher/term_results.html', {'used_terms': used_terms})
        images_results_template = render_to_string('searcher/results.html')

        context = {'term_results' : term_results_template, 'image_results' : images_results_template}
        return JsonResponse(context)




