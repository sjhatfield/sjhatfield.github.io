layout: archive
permalink: /machine-learning/
title: "Machine Learning Posts by Tags"
author_profile: true

---
{% include base_path %}
{% include group-by-array %}
colletion=site.posts="tages" %}

{% for tag in group_names %}
	{% assign posts =
	group_items[forloop.index0] %}
	<h2 id="{{ tag | slugify }}"
	class="archive__subtitle">{{ tag }} </h2>h2>
	{% for post in posts %}
		{% include archive-single.html %}
	{% endfor %}
{% endfor %}